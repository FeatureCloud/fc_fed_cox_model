import pickle
import time
import queue as q
import redis
import rq
from flask import Blueprint, jsonify, request, current_app
from lifelines.utils import (check_nans_or_infs,normalize,StepSizer,ConvergenceError,inv_normal_cdf,_get_index)
import pandas as pd
from collections import OrderedDict
import numpy as np
from scipy.linalg import solve as spsolve, LinAlgError, norm
import json
from scipy import stats
from lifelines.utils.concordance import concordance_index
from autograd import numpy as anp
from autograd import elementwise_grad


pool = redis.BlockingConnectionPool(host='localhost', port=6379, db=0, queue_class=q.Queue)
r = redis.Redis(connection_pool=pool)

## INITIALIZATION OF SOME PARAMETER
r.set('available', pickle.dumps(False))
r.set('step', pickle.dumps('start'))

r.set('local_norm', pickle.dumps(True))
r.set('global_norm', pickle.dumps(True))

r.set('local_init', pickle.dumps(True))
r.set('global_init', pickle.dumps(True))

r.set('local_stat', pickle.dumps(True))
r.set('global_stat', pickle.dumps(True))

r.set('local_c', pickle.dumps(True))
r.set('global_c', pickle.dumps(True))

r.set('master_step',pickle.dumps('master_upload'))
r.set('slave_step',pickle.dumps('slave_upload'))

r.set('iteration',pickle.dumps(0))
r.set('converging',pickle.dumps(True))

api_bp = Blueprint('api', __name__)
tasks = rq.Queue('fc_tasks', connection=r)

CONVERGENCE_DOCS = "Please see the following tips in the lifelines documentation: https://lifelines.readthedocs.io/en/latest/Examples.html#problems-with-convergence-in-the-cox-proportional-hazard-model"


@api_bp.route('/status', methods=['GET'])
def status():
    """
    GET request to /status, if True is returned a GET data request will be send
    :return: JSON with key 'available' and value True (/data GET request) or False (/data POST request)

    """
    current_app.logger.info('[API] /status GET request')
    cur_step = redis_get( 'step' )
    available = redis_get('available')
    converging = redis_get('converging')

    current_app.logger.info(f'[API] step: {cur_step}')
    current_app.logger.info(f'[API] converging: {converging}')
    current_app.logger.info(f'[API] available: {available}')

    if cur_step=="local_variance":
        # (slave and master: available = False)
        local_variance_calculation()

    elif cur_step=='normalization':
        # (slave and master: available = False)
        normalization()

    elif cur_step=='local_stat':
        # (slave and master: available = False)
        if (redis_get('master')):
            if (redis_get('iteration')==0):
                start_time = redis_get( 'start_time' )
                stop_time = time.time()
                current_app.logger.info(
                    f'[WEB] time for initializing the statistics, model parameter, step_size and so...: {stop_time- start_time}' )
                redis_set( 'start_time', stop_time )
                redis_set('iteration_time',stop_time)
                redis_set('update_time',stop_time)
            else:
                start_time = redis_get('iteration_time')
                stop_time = time.time()
                current_app.logger.info(
                    f'[WEB] time for one iteration: {stop_time- start_time}' )
                redis_set( 'iteration_time', stop_time )

                start_time = redis_get( 'update_time' )
                stop_time = time.time()
                current_app.logger.info(
                    f'[WEB] time sending updated model parameter: {stop_time- start_time}' )
                redis_set( 'update_time', stop_time )

        local_statistics_calculation()

    elif cur_step=='local_c_index':
        # (slave and master: available = False)

        if (redis_get('master')):
            start_time = redis_get( 'iteration_time' )
            stop_time = time.time()
            current_app.logger.info(
                f'[WEB] time for last iteration: {stop_time- start_time}' )

            start_time = redis_get( 'update_time' )
            stop_time = time.time()
            current_app.logger.info(
                f'[WEB] time sending updated model parameter (last iteration): {stop_time- start_time}' )
            redis_set( 'update_time', stop_time )

            start_time = redis_get( 'start_time' )
            stop_time = time.time()
            current_app.logger.info(
                f'[WEB] time for calculating the model parameter (all iterations): {stop_time- start_time}' )
            redis_set( 'start_time', stop_time )

        local_concordance_calculation()
        redis_set( 'master_step', 'master_final' )
        redis_set( 'slave_step', 'slave_final' )

    return jsonify({'available': True if available else False})

@api_bp.route('/data', methods=['GET', 'POST'])
def data():
    """
    GET request to /data sends data
    POST request to /data pulls data
    :return: GET request: JSON with key 'data' and value data
             POST request: JSON True
    """

    master = redis_get('master')
    master_step = redis_get( 'master_step' )
    slave_step = redis_get( 'slave_step' )
    step = redis_get('step')

    current_app.logger.info('[API] /data request')
    current_app.logger.info(f'[API] step: {step}')

    # data will be pulled from flask object request as json format
    if request.method == 'POST':

        if master:

            # get covariates of slaves to find intersection
            if master_step == 'master_intersect':
                current_app.logger.info( '[API] /data master_intersect POST request ' )
                local_covariates_json = request.get_json(True)['covariates']
                local_covariates = pd.read_json(local_covariates_json,typ='series',orient='records')
                global_cov = redis_get( 'global_cov' )
                global_cov.append( local_covariates )
                redis_set( 'global_cov', global_cov )
                global_covariates_intersection()
                return jsonify(True)

            # get local mean of slaves
            if master_step=='master_norm':
                current_app.logger.info( '[API] /data master_norm POST request ' )
                norm_step = redis_get("norm_step")

                if norm_step=="mean":
                    global_norm = redis_get('global_norm')
                    data_append = request.get_json(True)['local_norm']
                    mean_json = data_append[0]
                    n = data_append[1] # number of samples
                    mean = pd.read_json(mean_json,typ='series',orient='records')
                    global_norm.append((mean,n))
                    redis_set('global_norm', global_norm)
                    global_mean_calculation()

                elif norm_step=="variance":
                    global_norm = redis_get('global_norm')
                    data_append = request.get_json( True )[ 'local_norm' ]
                    variance_json = data_append
                    mean = pd.read_json( variance_json, typ='series', orient='records' )
                    global_norm.append( (mean) )
                    redis_set( 'global_norm', global_norm )
                    global_variance_calculation()

                return jsonify(True)

            # get initialized local statistics of slaves
            elif master_step=='master_init':
                current_app.logger.info( '[API] /data master_init POST request ' )
                global_init = redis_get( 'global_init' )
                data_append = request.get_json( True )[ 'local_init' ]
                zlr_json = data_append[ 0 ]
                distinct_times_json = data_append[ 1 ]
                count_d = data_append[2]
                zlr = pd.read_json(zlr_json, typ='series', orient='records' )
                distinct_times = pd.read_json( distinct_times_json, typ='series', orient='records' )
                global_init.append( (zlr,distinct_times,count_d ) )
                redis_set( 'global_init', global_init )
                redis_set('step','global_init')
                global_initialization()
                return jsonify( True )

            # get local aggregated statistics for each iteration of slaves
            elif master_step=='master_stat':
                current_app.logger.info( '[API] /data master_stat POST request ' )
                global_stat = redis_get('global_stat')
                data_append = request.get_json( True )[ 'local_stat' ]
                i1 = data_append[ 0 ]
                i2 = data_append[ 1 ]
                i3 = data_append[ 2 ]
                global_stat.append( (i1, i2, i3) )
                redis_set( 'global_stat', global_stat )
                redis_set('step','update_beta')
                global_beta_calculation()
                return jsonify( True )

            # get local concordance index of the slaves
            elif master_step=='master_final':
                current_app.logger.info( '[API] /data master_final POST request ' )
                global_c = redis_get('global_c')
                c_index = request.get_json(True)['local_c']
                global_c.append(c_index)
                redis_set('global_c',global_c)
                redis_set('step','summary')
                global_summary_calculation()
                return jsonify(True)


        else:
            # get parameter of the cox model of master to display in the start_client.html
            if slave_step == 'slave_upload':
                current_app.logger.info( '[API] /data slave_upload POST request ' )
                upload_info = request.get_json( True )[ 'upload_info' ]
                max_steps = upload_info[0]
                precision = upload_info[1]
                duration_col = upload_info[2]
                event_col = upload_info[3]
                covariates_json = upload_info[4]
                covariates = json.loads(covariates_json)
                l1_ratio = upload_info[5]
                penalization = upload_info[6]
                intersection = upload_info[7]

                redis_set('max_steps',max_steps)
                redis_set('precision',precision)
                redis_set('duration_col',duration_col)
                redis_set('event_col',event_col)
                redis_set('master_covariates',covariates)
                redis_set('l1_ratio',l1_ratio)
                redis_set('penalization',penalization)
                redis_set('intersection',intersection)

                redis_set('available',False)

                redis_set('step','setup_slave')

                return jsonify(True)

            # get intersected covariates from master
            elif slave_step == 'slave_intersect':
                current_app.logger.info( '[API] /data slave_intersect POST request ' )
                covariates_json = request.get_json(True)['intersected_covariates']
                covariates = pd.read_json(covariates_json,typ='series',orient='records')
                redis_set('intersected_covariates',covariates)
                redis_set( 'slave_step', 'slave_norm' )
                redis_set( 'master_step', 'master_norm' )

                redis_set('step','local_mean')
                local_mean_calculation()

                return (jsonify(True))

            # get global mean from master for further calculations
            elif slave_step == 'slave_norm':
                current_app.logger.info( '[API] /data slave_norm POST request ' )
                norm_step = redis_get("norm_step")

                if norm_step=="mean":
                    global_mean = request.get_json(True)['global_mean']
                    mean = pd.read_json( global_mean, typ='series', orient='records' )
                    redis_set('global_mean',mean)
                    redis_set('step','local_variance')

                if norm_step=="variance":
                    global_std = request.get_json(True)['global_std']
                    std = pd.read_json( global_std, typ='series', orient='records' )
                    redis_set( 'global_std', std )
                    redis_set('step','normalization')

                return jsonify(True)

            # get initialized model parameter (zeros)
            elif slave_step == 'slave_init':
                current_app.logger.info( '[API] /data slave_init POST request ' )
                beta_0_json = request.get_json(True)['beta_0']
                beta_0 = json.loads(beta_0_json)
                redis_set('beta',beta_0)
                redis_set('step','local_stat')
                return jsonify(True)

            # get updated model parameter and converging criteria
            elif slave_step == 'slave_stat':
                current_app.logger.info( '[API] /data slave_stat POST request ' )
                data = request.get_json( True )[ 'beta' ]
                beta = json.loads( data[0] )
                converging = data[1]
                redis_set('converging',converging)
                redis_set( 'beta', beta)
                if converging:
                    #step_to_local_stat() #new iteration -> step: local_stat
                    redis_set('step','local_stat')
                    redis_set( 'available', False )
                else:
                    redis_set('step','local_c_index')
                    redis_set('available',False) # slaves & master because no further data exchange
                return jsonify(True)

            # get result dataframe and global concordance index from master
            elif slave_step == 'slave_final':
                current_app.logger.info( '[API] /data slave_final POST request ' )
                result = request.get_json( True )[ 'result' ]
                summary_json = result[0]
                c_index = result[1]
                summary = pd.read_json(summary_json,typ='frame',orient='records')
                redis_set('result',summary)
                redis_set('c_index',c_index)
                redis_set('step','final') # final.html will be displayed with summary dataframe

                return jsonify(True)

    # data will be send to the master
    elif request.method == 'GET':
        if master:
            # send parameter for cox regression to slaves
            if master_step == 'master_upload':
                current_app.logger.info( '[API] /data master_upload GET request ' )
                max_steps = redis_get('max_steps')
                precision = redis_get('precision')
                duration_col = redis_get('duration_col')
                event_col = redis_get('event_col')
                l1_ratio = redis_get('l1_ratio')
                penalization = redis_get('penalization')
                intersection = redis_get('intersection')

                data = redis_get('data')
                covariates = data.columns.values
                index = np.argwhere( covariates == duration_col )
                covariates = np.delete(covariates,index)
                index = np.argwhere( covariates == event_col )
                covariates = np.delete(covariates,index)
                redis_set('master_covariates',covariates)
                covariates_json = covariates.tolist()
                covariates_json = json.dumps(covariates_json)

                redis_set('available',False)
                redis_set( 'master_step', 'master_intersect' )
                redis_set( 'slave_step', 'slave_intersect' )

                return jsonify({'upload_info':(max_steps,precision,duration_col,event_col,covariates_json,l1_ratio,penalization,intersection)})

            # send the intersection of all covariates to slaves
            elif master_step == 'master_intersect':
                current_app.logger.info( '[API] /data master_norm GET request ' )
                intersected_covariates = redis_get('intersected_covariates')
                covariates_json = intersected_covariates.to_json()
                redis_set( 'available', False )
                redis_set( 'slave_step', 'slave_norm' )
                redis_set( 'master_step', 'master_norm' )
                redis_set('step','local_mean')
                local_mean_calculation()
                return jsonify({'intersected_covariates':(covariates_json)})

            # send global mean to slaves
            elif master_step == 'master_norm':
                current_app.logger.info( '[API] /data master_norm GET request ' )
                norm_step = redis_get("norm_step")
                if norm_step=="mean":
                    global_mean = redis_get('global_mean')
                    mean_json = global_mean.to_json(double_precision=15)
                    redis_set('available',False)
                    redis_set('step','local_variance')
                    return jsonify( {'global_mean': (mean_json)} )

                elif norm_step=="variance":
                    global_std = redis_get( 'global_std' )
                    std_json = global_std.to_json( double_precision=15 )
                    redis_set( 'available', False )
                    redis_set( 'step', 'normalization' )

                    return jsonify( {'global_std': (std_json)} )

            # send initial model parameter to slaves (zeros)
            elif master_step == 'master_init':
                current_app.logger.info( '[API] /data master_init GET request ' )
                zr = redis_get( 'global_zr' )
                covariates = zr.axes[ 0 ]

                #initialize model parameter
                beta_0 = np.zeros((len(covariates),))
                beta_0_json = beta_0.tolist()
                beta_0_json = json.dumps(beta_0_json)
                redis_set('beta',beta_0)

                # initialize step_sizer
                step_size = None
                step_sizer = StepSizer(step_size)
                step_size = step_sizer.next()
                redis_set('step_sizer',step_sizer)
                redis_set('step_size',step_size)

                #initialize delta
                delta = np.zeros_like(beta_0)
                redis_set('delta',delta)

                redis_set('available',False)
                redis_set('step','local_stat')
                return jsonify( {'beta_0': (beta_0_json)} )

            # send updated model parameter and convergence criteria to slaves
            elif master_step == 'master_stat':
                current_app.logger.info( '[API] /data master_stat GET request ' )
                beta = redis_get('beta')
                converging = redis_get('converging')
                beta_json = beta.tolist()
                beta_json = json.dumps( beta_json )
                redis_set( 'available', False ) # in /status local_statistics_calculation will be called
                if converging:
                    redis_set('step','local_stat')
                else:
                    redis_set('step','local_c_index') # -> summary

                return jsonify({'beta':(beta_json,converging)})

            # send summary results and concordance index to slaves
            elif master_step == 'master_final':
                current_app.logger.info( '[API] /data master_final GET request ' )
                c_index = redis_get('c_index')
                result = redis_get('result')
                result_json = result.to_json()
                redis_set( 'available', False )

                start_time = redis_get( 'start_time' )
                stop_time = time.time()
                current_app.logger.info(
                    f'[WEB] time for calculating the result dataframe and concordance index: {stop_time- start_time}' )
                redis_set( 'start_time', stop_time )

                start_time = redis_get( 'global_time' )
                stop_time = time.time()
                current_app.logger.info(
                    f'[WEB] time for whole Cox model: {stop_time- start_time}' )

                redis_set('step','final')
                return jsonify({'result':(result_json,c_index)})

        else:

            # send covariates to master to check the intersection of the covariates
            if slave_step == 'slave_intersect':
                current_app.logger.info( '[API] /data slave_intersect GET request ' )
                covariates = redis_get('covariates')
                redis_set( 'available', False )
                return jsonify({'covariates':(covariates.to_json())})

            # send local mean to master
            elif slave_step == 'slave_norm':
                current_app.logger.info( '[API] /data slave_norm GET request ' )
                norm_step = redis_get('norm_step')

                if norm_step=="mean":
                    local_norm = redis_get('local_norm')
                    current_app.logger.info(local_norm)
                    mean = local_norm[0]
                    n = local_norm[1] # number of samples
                    mean_json = mean.to_json(double_precision=15)
                    redis_set('available', False)
                    return jsonify({'local_norm':(mean_json,n)})

                elif norm_step=="variance":
                    local_norm = redis_get( 'local_norm' )
                    current_app.logger.info( local_norm )
                    variance = local_norm
                    variance_json = variance.to_json( double_precision=15 )
                    redis_set( 'available', False )
                    return jsonify( {'local_norm': (variance_json)} )

            # send initial statistics to master
            elif slave_step=='slave_init':
                current_app.logger.info( '[API] /data slave_init GET request ' )
                local_init = redis_get('local_init')
                current_app.logger.info( local_init )
                zlr = local_init[0]
                distinct_times = local_init[1]
                numb_d_set = local_init[2]
                zlr_json = zlr.to_json(double_precision=15)
                distinct_times_json = distinct_times.to_json(double_precision=15)
                redis_set('available',False) # POST -> clients will get the initial model parameter
                redis_set('step','global_init')
                return jsonify({'local_init':(zlr_json,distinct_times_json,numb_d_set)})

            # send updated aggregated statistics to master
            elif slave_step=='slave_stat':
                current_app.logger.info( '[API] /data slave_stat GET request ' )
                local_stat = redis_get('local_stat')
                current_app.logger.info(local_stat)
                i1 = local_stat[0]
                i2 = local_stat[1]
                i3 = local_stat[2]
                redis_set('step','send_model_parameter')
                redis_set( 'available', False )  # POST -> clients will get the new model parameter
                return jsonify( {'local_stat': (i1, i2, i3)} )

            # send local concordance index to master
            elif slave_step=='slave_final':
                current_app.logger.info( '[API] /data slave_final GET request ' )
                c_index = redis_get('local_c')
                current_app.logger.info(c_index)
                redis_set('available',False)
                redis_set('step','summary')
                return jsonify({'local_c':(c_index)})

    else:
        current_app.logger.info('[API] Wrong request type, only GET and POST allowed')
        return jsonify(True)

@api_bp.route('/setup', methods=['POST'])
def setup():
    """
    set setup params, id is the id of the client, master is True if the client is the master,
    in global_data the data from all clients (including the master) will be aggregated,
    clients is a list of all ids from all clients, nr_clients is the number of clients involved in the app
    :return: JSON True
    """
    current_app.logger.info('[API] SETUP')
    setup_params = request.get_json()
    current_app.logger.info(setup_params)
    redis_set('id', setup_params['id'])
    master = setup_params['master']
    redis_set('master', master)
    if master:
        redis_set('global_cov',[])
        redis_set('global_norm', [])
        redis_set('global_init',[])
        redis_set( 'global_stat', [])
        redis_set( 'global_c', [ ] )
    redis_set('clients', setup_params['clients'])
    redis_set('nr_clients', len(setup_params['clients']))
    redis_set('step','setup_master')
    current_app.logger.info('[API] Setup parameters saved')
    return jsonify(True)

def redis_get(key):
    if key in r:
        return pickle.loads(r.get(key))
    else:
        return None

def redis_set(key, value):
    r.set(key, pickle.dumps(value))

def preprocessing():
    """
    Preprocesses the dataframe of the clients. Save X, T and E of the dataset.
    """
    current_app.logger.info('[API] run preprocessing')
    localdata = redis_get('data')
    duration_col = redis_get('duration_col')
    event_col = redis_get('event_col')

    if localdata is None:
        current_app.logger.info('[API] Data is None')
        return None
    elif duration_col is None:
        current_app.logger.info( '[API] Duration column is None')
        return None
    elif event_col is None:
        current_app.logger.info( '[API] Event column is None')
        return None
    else:
        sort_by = [duration_col, event_col] if event_col else [duration_col]
        data = localdata.sort_values(by=sort_by)
        T = data.pop(duration_col)
        E = data.pop(event_col)

        X = data.astype(float)
        T = T.astype(float)

        check_nans_or_infs(E)
        E = E.astype(bool)

        redis_set('X',X)
        redis_set('T',T)
        redis_set('E',E)

        current_app.logger.info( '[API] Preprocessing is done (X,T,E saved)' )

        redis_set('step','find_intersection')

        get_local_covariates()

def get_local_covariates():
    """
    get the name of the covariates of the local dataframe and send to the master
    """
    current_app.logger.info( '[API] run get_local_covariates' )

    X = redis_get( 'X' )
    duration_col = redis_get('duration_col')
    event_col = redis_get('event_col')
    covariates = X.columns.values
    index = np.argwhere( covariates == duration_col )
    covariates = np.delete( covariates, index )
    index = np.argwhere( covariates == event_col )
    covariates = np.delete( covariates, index )
    redis_set('covariates',pd.Series(covariates))
    #current_app.logger.info(f'[API] covariates: {covariates}')
    if redis_get( 'master' ):
        global_cov = redis_get( 'global_cov' )
        global_cov.append(pd.Series(covariates))
        redis_set( 'global_cov', global_cov )
        global_covariates_intersection()

    else:
        redis_set( 'available', True)

def global_covariates_intersection():
    current_app.logger.info( '[API] run global_covariates_intersection' )
    global_cov = redis_get( 'global_cov' )
    nr_clients = redis_get( 'nr_clients' )
    np.set_printoptions( precision=30 )
    if len( global_cov ) == nr_clients:
        current_app.logger.info( '[API] The data of all clients has arrived' )

        start_time = time.time() #start of runtime check
        redis_set('start_time',start_time)
        redis_set('global_time',start_time)

        intersect_covariates = pd.Series()

        for cov in global_cov:
            if intersect_covariates.empty:
                intersect_covariates = cov
            else:
                intersect_covariates = pd.Series(np.intersect1d(intersect_covariates,cov))

        #current_app.logger.info( f'[API] intersection of covariates: {intersect_covariates}' )
        redis_set( 'intersected_covariates', intersect_covariates )
        redis_set( 'available', True ) # send the intersected covariates to slaves

    else:
        current_app.logger.info( '[API] Not the data of all clients has been send to the master' )

def local_mean_calculation():
    """
    Udate the dataset which is used for the cox regression using the intersection of covariates and
    calculate mean of the local dataframe and send to the master

    """
    if redis_get('master'):
        start_time = redis_get( 'start_time' )
        stop_time = time.time()
        current_app.logger.info( f'[WEB] time for covariates intersection calculation and sending: {stop_time- start_time}' )
        redis_set( 'start_time', stop_time )

    current_app.logger.info('[API] run local_mean_calculation')
    X = redis_get('X')
    if X is None:
        current_app.logger.info('[API] X is None')
        return None
    else:
        # update X
        intersected_covariates = redis_get('intersected_covariates')
        X = X[intersected_covariates.tolist()]
        redis_set('X',X)

        # start normalization
        norm_mean = X.mean(0)
        n,d = X.shape #needed for penalization

        if redis_get('master'):
            redis_set("norm_step","mean")
            redis_set('step','send_to_master')
            global_norm = redis_get('global_norm')
            global_norm.append((norm_mean,n))
            redis_set('global_norm', global_norm)
            global_mean_calculation()
        else:
            redis_set("norm_step","mean")
            redis_set('local_norm',(norm_mean,n))
            redis_set('step','send_to_master')
            redis_set('available', True)

def global_mean_calculation():
    """
    calculate the global mean if the data of all clients arrived
    """
    current_app.logger.info('[API] run global_mean_calculation')
    global_norm = redis_get('global_norm')
    nr_clients = redis_get('nr_clients')
    np.set_printoptions( precision=30 )
    if len(global_norm) == nr_clients:
        current_app.logger.info('[API] The data of all clients has arrived')
        mean = pd.Series()
        n=0 # number of samples

        for client in global_norm:
            mean = mean.add(client[0],fill_value=0)
            n += client[1]

        result_mean = mean/nr_clients
        np.set_printoptions(precision=30)
        current_app.logger.info( f'[API] global mean result: {result_mean.to_numpy()}')
        current_app.logger.info(f'[API] nr_samples: {n}')
        redis_set('global_mean', result_mean)
        redis_set('nr_samples',n)
        redis_set( 'available', True )
        redis_set('step','global_mean')
    else:
        current_app.logger.info('[API] Not the data of all clients has been send to the master')

def local_variance_calculation():
    """
        calculate variance of the local dataframe and send to the master

        """

    # calculate local variance using the global mean
    global_mean = redis_get("global_mean")
    X = redis_get("X")
    variance = np.mean((X-global_mean)**2,axis=0)

    if redis_get( 'master' ):
        redis_set('norm_step',"variance")
        redis_set( 'step', 'send_to_master' )
        global_norm = []
        global_norm.append( (variance) )
        redis_set( 'global_norm', global_norm )
        global_variance_calculation()
    else:
        redis_set( 'norm_step', "variance" )
        redis_set( 'local_norm', (variance) )
        redis_set( 'step', 'send_to_master' )
        redis_set( 'available', True )

def global_variance_calculation():
    """
        calculate the global variance and std if the data of all clients arrived
        """
    current_app.logger.info( '[API] run global_variance_calculation' )
    global_norm = redis_get( 'global_norm' )
    nr_clients = redis_get( 'nr_clients' )
    np.set_printoptions( precision=30 )
    if len( global_norm ) == nr_clients:
        current_app.logger.info( '[API] The data of all clients has arrived' )
        variance=pd.Series()
        for client in global_norm:
            variance = variance.add( client, fill_value=0 )

        result_variance = variance / nr_clients
        result_std = np.sqrt(result_variance)

        np.set_printoptions( precision=30 )
        current_app.logger.info( f'[API] global std result: {result_std.to_numpy()}' )
        redis_set( 'global_std', result_std )
        redis_set( 'available', True )
        redis_set( 'step', 'global_variance' )
    else:
        current_app.logger.info( '[API] Not the data of all clients has been send to the master' )

def normalization():
    """
    normalize dataset of client with global_mean and global_std
    """
    current_app.logger.info( '[API] run normalization of each client' )

    mean = redis_get('global_mean')
    std = redis_get('global_std')

    X = redis_get('X')
    X_norm = pd.DataFrame(normalize(X.values,mean.values,std.values),index=X.index,columns=X.columns)
    redis_set('X_norm',X_norm)

    redis_set('master_step','master_init')
    redis_set('slave_step','slave_init')
    redis_set('step','local_init')

    if redis_get('master'):
        start_time = redis_get( 'start_time' )
        stop_time = time.time()
        current_app.logger.info( f'[WEB] time for normalization of dataset: {stop_time- start_time}' )
        redis_set( 'start_time', stop_time )

    local_initialization()

def local_initialization():
    """
    calculate statistics D, zlr, and count_d of each client
    """
    current_app.logger.info( '[API] run local_initialization' )
    duration_col = redis_get('duration_col')
    event_col = redis_get('event_col')
    # sort dataframe by duration_column
    sort_by = [duration_col]
    data = redis_get('data')
    data = data.sort_values(by=sort_by)
    #risk_set = {}
    death_set = {}
    numb_d_set = {}

    # get the distinct event times
    distinct_times = pd.Series(data[ duration_col ].unique() )

    for uniq_time in distinct_times:
        #Ri = data[ data[ duration_col ] >= uniq_time ].index.tolist()
        Di = data[(data[ duration_col ] == uniq_time) & (data[ event_col ] == 1) ].index.tolist()
        #risk_set[ uniq_time ] = Ri
        death_set[ uniq_time ] = Di
        numb_d_set[ str(uniq_time) ] = str(len( Di ))

    # calculate z-value (zlr) // sum over all distinct_times (sum of the covariates over all individuals in Di))
    zlr = pd.Series()
    X_norm = redis_get('X_norm')
    for time in distinct_times:
        covariates = X_norm.loc[ death_set[ time ], ]
        sum = covariates.sum( axis=0, skipna=True )
        zlr = zlr.add( sum, fill_value=0 )

    if redis_get( 'master' ):
        redis_set('step','send_to_master')
        global_init = redis_get( 'global_init' )
        global_init.append( (zlr, distinct_times,numb_d_set) )
        redis_set( 'global_init', global_init )
        global_initialization()
    else:
        redis_set( 'local_init', (zlr, distinct_times,numb_d_set) )
        redis_set('step','send_to_master')
        redis_set( 'available', True )

def global_initialization():
    """
    global initialization of the server: zlr, D, count_d

    """
    current_app.logger.info( '[API] run global_initialization' )
    global_init = redis_get( 'global_init' )
    nr_clients = redis_get( 'nr_clients' )
    if (len(global_init))==nr_clients:
        D = []
        zr = pd.Series()
        for client in global_init:
            zlr = client[0]
            distinct_times = client[1]
            D.extend(distinct_times.tolist())
            zr = zr.add(zlr, fill_value=0)

        # delete duplicates out of D to get the total number of distinct event times
        D = list( OrderedDict.fromkeys( D ) )
        D.sort()

        # sum over all sites |Dki| ({time i : |Dki|}
        count_d = {}
        for t in D:
            val = 0
            for client in global_init:
                n = client[2].get(str(t))
                if n is not None:
                    val = val + int(n)
            count_d[ t ] = val

        redis_set('global_count_d',count_d)
        redis_set('global_D',D)
        redis_set('global_zr',zr)
        #current_app.logger.info( f'[API] global D: {D}' )
        #current_app.logger.info( f'[API] global zr: {zr.values}' )
        #current_app.logger.info( f'[API] global count_d: {count_d}' )

        redis_set( 'available', True )  # master will afterwards send the initial model parameter to the clients (GET)

    else:
        current_app.logger.info( '[API] Not the data of all clients has been send to the master' )

def local_statistics_calculation():

    """
    calculate three local aggregated statistics using the parameter beta.
    send them to the server.
    """

    current_app.logger.info( '[API] run local_statistics' )

    np.set_printoptions( precision=30 )

    i1 ={}
    i2 ={}
    i3 ={}

    X_norm = redis_get('X_norm')
    T = redis_get('T')
    beta = redis_get('beta')

    X = X_norm.values
    T = T.values

    n,d = X.shape
    risk_phi = 0
    risk_phi_x = np.zeros((d,))
    risk_phi_x_x = np.zeros((d,d))


    _, counts = np.unique(-T,return_counts=True)
    scores = np.exp(np.dot(X,beta))
    pos=n
    time_index=0
    for count_of_removal in counts:
        uniq_time = _[time_index]
        slice_ = slice(pos-count_of_removal,pos)
        X_at_t = X[slice_]

        phi_i = scores[slice_,None]
        phi_x_i = phi_i * X_at_t
        phi_x_x_i = np.dot(X_at_t.T,phi_x_i)

        risk_phi = risk_phi + phi_i.sum()
        risk_phi_x = risk_phi_x + (phi_x_i).sum(0)
        risk_phi_x_x = risk_phi_x_x + phi_x_x_i

        risk_phi_x_json = risk_phi_x.tolist()
        risk_phi_x_json = json.dumps( risk_phi_x_json )

        risk_phi_x_x_json = risk_phi_x_x.tolist()
        risk_phi_x_x_json = json.dumps( risk_phi_x_x_json )

        pos = pos-count_of_removal
        t = str(-uniq_time)
        i1[t] = risk_phi
        i2[t] = risk_phi_x_json
        i3[t] = risk_phi_x_x_json

        time_index+=1

    redis_set('master_step','master_stat')
    redis_set('slave_step','slave_stat')

    if redis_get( 'master' ):
        global_stat = redis_get( 'global_stat' )
        global_stat.append( (i1, i2,i3) )
        redis_set( 'global_stat', global_stat )
        redis_set('available',False) #POST -> master will get the aggregated statistics from slaves
        redis_set('step','send_to_master')

        start_time = redis_get( 'update_time' )
        stop_time = time.time()
        current_app.logger.info(
            f'[WEB] time for local statistics calculation: {stop_time- start_time}' )
        redis_set( 'update_time', stop_time )

        global_beta_calculation()
    else:
        redis_set( 'local_stat', (i1, i2,i3) )
        redis_set( 'available', True ) #GET -> slaves will send aggregated local statistics to master
        redis_set( 'step', 'send_to_master' )

def global_beta_calculation():
    """
    Update the model parameter beta and hessian.
    """
    current_app.logger.info( '[API] run global_beta_calculation' )
    global_stat = redis_get( 'global_stat' )
    beta = redis_get('beta')
    nr_clients = redis_get( 'nr_clients' )

    converging = redis_get('converging')
    step_sizer = redis_get( 'step_sizer' )
    step_size = redis_get('step_size')
    delta = redis_get('delta')

    precision = redis_get('precision')
    max_steps = redis_get('max_steps')
    iteration = redis_get('iteration')

    penalization = redis_get('penalization')
    l1_ratio = redis_get('l1_ratio')

    n = redis_get('nr_samples')

    soft_abs = lambda x, a: 1 / a * (anp.logaddexp( 0, -a * x ) + anp.logaddexp( 0, a * x ))
    penalizer = (
        lambda beta, a: n * 0.5
                        * penalization
                        * (l1_ratio * soft_abs( beta, a ).sum() + (1 - l1_ratio) * (
                (beta) ** 2).sum())
    )
    d_penalizer = elementwise_grad( penalizer )
    dd_penalizer = elementwise_grad( d_penalizer )

    covariates = redis_get('global_zr').axes[ 0 ]
    d = len( covariates )

    if (len( global_stat )) == nr_clients:

        start_time = redis_get( 'update_time' )
        stop_time = time.time()
        current_app.logger.info(
            f'[WEB] time for sending local statistics to server: {stop_time- start_time}' )
        redis_set( 'update_time', stop_time )

        iteration = iteration + 1
        h,g = get_efron_values(global_stat)

        if penalization > 0:
            g -= d_penalizer( beta, 1.3 ** iteration )
            h[ np.diag_indices( d ) ] -= dd_penalizer( beta, 1.3 ** iteration )

        try:
            inv_h_dot_g = spsolve(-h,g,assume_a="pos",check_finite=False)
        except (ValueError, LinAlgError) as e:
            if "infs or NaNs" in str(e):
                raise ConvergenceError(
                    """Hessian or gradient contains nan or inf value(s). Convergence halted. {0}""".format(CONVERGENCE_DOCS),
                    e,
                )
            elif isinstance(e, LinAlgError):
                raise ConvergenceError(
                    """Convergence halted due to matrix inversion problems. Suspicion is high collinearity. {0}""".format(
                        CONVERGENCE_DOCS
                    ),
                    e,
                )
            else:
                # something else?
                raise e

        delta = inv_h_dot_g
        hessian, gradient = h,g
        if delta.size > 0:
            norm_delta = norm(delta)
        else:
            norm_delta = 0

        #current_app.logger.info(f'[API] norm_delta:{norm_delta}')

        newton_decrement = g.dot(inv_h_dot_g)/2

        #convergence criteria
        if norm_delta < precision:
            converging, success = False, True
        # this is what R uses by default
        elif newton_decrement < precision:
            converging, success = False, True
        #maximal number of iterations reached
        elif iteration>=max_steps:
            converging, success = False, False
        elif step_size <= 0.00001:
            converging, success = False, False

        step_size = step_sizer.update(norm_delta).next()
        redis_set('step_size',step_size)
        redis_set('step_sizer',step_sizer)
        redis_set('converging',converging)

        beta += step_size * delta
        #current_app.logger.info( f'[API] new updated beta: {beta}' )

        redis_set('beta',beta)
        redis_set('hessian',hessian)

        # set iteration for penalizer
        redis_set('iteration',iteration)

        redis_set('global_stat',[]) # clear for new iteration

        redis_set( 'step', 'send_model_parameter' )
        redis_set( 'available', True ) # send model parameter to slaves

        start_time = redis_get( 'update_time' )
        stop_time = time.time()
        current_app.logger.info(
            f'[WEB] time for updating model parameter: {stop_time- start_time}' )
        redis_set( 'update_time', stop_time )

    else:
        current_app.logger.info( '[API] Not the data of all clients has been send to the master' )

def get_efron_values(global_stat):
    """
    :param global_stat: aggregated statistics of all clients
    Calculate the hessian and gradient out of the aggregated statistics from the slaves.
    """

    current_app.logger.info( '[API] run get_efron_values' )

    zr = redis_get( 'global_zr' )
    covariates = zr.axes[ 0 ]
    d = len( covariates )

    global_i1 = {}
    global_i2 = {}
    global_i3 = {}

    for client in global_stat:

        last_i1 = 0
        last_i2 = np.zeros( (d,) )
        last_i3 = np.zeros( (d, d) )

        # send sites an updated beta and sites calculate aggregated statistics
        i1 = client[ 0 ]
        i2 = client[ 1 ]
        i3 = client[ 2 ]

        D = redis_get( 'global_D' )
        count_d = redis_get( 'global_count_d' )

        for time in sorted( D, reverse=True ):
            np.set_printoptions( precision=30 )

            t = str(float(time)) # in D there are ints but json makes out of integer keys -> string keys and in i1 there are strings of float numbers
            # if time already in i1 we can normally add the value of time to global_i1 or create a new key with the value in global_i1
            if t in i1:
                if t in global_i1:
                    df = global_i1[ t ]
                    global_i1[ t ] = df + i1[t]
                    i2_t = json.loads( i2[ t ] )
                    i2_t = np.array(i2_t)
                    df = global_i2[ t ]
                    global_i2[ t ] = df + i2_t
                    i3_t = json.loads( i3[ t ] )
                    i3_t = np.array( i3_t )
                    df = global_i3[ t ]
                    global_i3[ t ] = df + i3_t

                else:
                    i2_t = json.loads(i2[t])
                    i2_t = np.array( i2_t )
                    i3_t = json.loads(i3[t])
                    i3_t = np.array( i3_t )
                    global_i1[ t ] = i1[ t ]
                    global_i2[ t ] = i2_t
                    global_i3[ t ] = i3_t

                last_i1 = i1[ t ]
                last_i2 = i2_t
                last_i3 = i3_t
            # if time is not in i1 we have to add the value of the key time-1 of i1 to global_i1
            else:
                if t in global_i1:
                    global_i1[ t ] = global_i1[ t ] + last_i1
                    global_i2[ t ] = global_i2[ t ] + last_i2
                    global_i3[ t ] = global_i3[ t ] + last_i3


                else:
                    global_i1[ t ] = last_i1
                    global_i2[ t ] = last_i2
                    global_i3[ t ] = last_i3


    # calculate first and second order derivative
    d1 = np.zeros( (d,) )
    d2 = np.zeros( (d, d) )

    for time in D:
        t = str(float(time))
        Dki = count_d[ time ]
        numer = global_i2[ t ] #in global_i1 and global_i2 -> keys are strings
        denom = 1.0 / np.array( [ global_i1[ t ] ] )
        summand = numer * denom[ :, None ]
        d1 = d1 + Dki * summand.sum( 0 )
        a1 = global_i3[ t ] * denom
        a2 = np.dot( summand.T, summand )
        d2 = d2 + Dki * (a2 - a1)

    # first order derivative
    zr = zr.to_numpy()
    gradient = zr - d1
    hessian = d2


    return hessian, gradient

def local_concordance_calculation():
    current_app.logger.info( '[API] run calculate concordance-index' )
    beta = redis_get( 'beta' )
    norm_std = redis_get( 'global_std' )
    params_ = beta / norm_std.values

    index_cov = redis_get('X').columns
    params_ = pd.Series( params_, index=index_cov, name="coef" )
    redis_set('params_',params_)

    T = redis_get('T')
    E = redis_get('E')
    X = redis_get('X')

    hazards = -_predict_partial_hazard(X,params_)

    c_index = concordance_index(T,hazards,E)

    if redis_get( 'master' ):
        redis_set('step','send_to_master')
        global_c = redis_get( 'global_c' )
        global_c.append( (c_index) )
        redis_set( 'global_c', global_c )
        global_summary_calculation()
    else:
        redis_set( 'local_c', (c_index) )
        redis_set('step','send_to_master')
        redis_set( 'available', True ) # will send the local concordance index to the master

def global_summary_calculation():
    """

    :return: summary of results & global concordance index

    This method calculates the model parameter, the global c-index and some additional results (hazard ratios, confidence intervals, ...).
    """
    current_app.logger.info('[API] run calculate final statistics')

    global_c = redis_get( 'global_c' )
    nr_clients = redis_get( 'nr_clients' )
    if (len( global_c )) == nr_clients:

        # calculate global_concordance index
        c_index = 0
        for local_index in global_c:
            c_index += local_index

        c_index = c_index / len(global_c)
        redis_set( 'c_index', c_index )

        current_app.logger.info( f'[API] concordance index: {c_index} ' )

        # prepare summary dataframe
        beta = redis_get('beta')
        norm_std = redis_get('global_std')
        params_ = beta / norm_std.values
        zr = redis_get('global_zr')
        hessian = redis_get('hessian')
        params_ = pd.Series( params_, index=zr.axes[ 0 ], name="coef" )

        #current_app.logger.info( f'[API] final model parameter: {params_} ' )

        hazard_ratios_ = pd.Series( np.exp(params_ ), index=zr.axes[ 0 ], name="exp(coef)" )
        alpha = 0.05
        variance_matrix_ = _calculate_variance_matrix( hessian, norm_std, zr)
        standard_errors_ = _calculate_standard_errors(variance_matrix_,params_)
        p_values_ = pd.Series(_calculate_p_values(params_,standard_errors_), index=zr.axes[ 0 ] )
        confidence_intervals_ = _calculate_confidence_intervals( alpha,params_,standard_errors_ )

        ci = 100 * (1 - alpha)
        z = inv_normal_cdf( 1 - alpha / 2 )
        with np.errstate( invalid="ignore", divide="ignore", over="ignore", under="ignore" ):
            df = pd.DataFrame( index=params_.index )
            df[ "coef" ] = params_
            df[ "exp(coef)" ] = hazard_ratios_
            df[ "se(coef)" ] = standard_errors_
            df[ "coef lower %g%%" % ci ] = confidence_intervals_[ "%g%% lower-bound" % ci ]
            df[ "coef upper %g%%" % ci ] = confidence_intervals_[ "%g%% upper-bound" % ci ]
            df[ "exp(coef) lower %g%%" % ci ] = hazard_ratios_ * np.exp( -z * standard_errors_ )
            df[ "exp(coef) upper %g%%" % ci ] = hazard_ratios_ * np.exp( z * standard_errors_ )
            df[ "z" ] = _calculate_z_values(params_,standard_errors_)
            df[ "p" ] = p_values_
            df[ "-log2(p)" ] = -np.log2( df[ "p" ] )

        redis_set('result',df)
        current_app.logger.info(f'[API] result dataframe: {df} ')
        # current_app.logger.info( f'[API] final model parameter: {params_} ' )

        # send summary dataframe and concordance_index to slaves
        redis_set('available',True)

    else:
        current_app.logger.info( '[API] Not the data of all clients has been send to the master' )

def _predict_partial_hazard(X,params_) -> pd.Series:
    """

    :return: partial_hazard

    """
    hazard = np.exp(_predict_log_partial_hazard(X,params_))
    return hazard

def _predict_log_partial_hazard(X,params_) -> pd.Series:
    hazard_names = params_.index
    norm_mean = redis_get('global_mean')

    if isinstance( X, pd.Series ) and (
            (X.shape[0] == len( hazard_names ) + 2) or (X.shape[0] == len( hazard_names ))):
        X = X.to_frame().T
        return _predict_log_partial_hazard(X,params_)
    elif isinstance( X, pd.Series ):
        assert len( hazard_names ) == 1, "Series not the correct argument"
        X = X.to_frame().T
        return _predict_log_partial_hazard(X,params_)

    index = _get_index( X )

    if isinstance( X, pd.DataFrame ):
        order = hazard_names
        X = X.reindex( order, axis="columns" )
        X = X.astype( float )
        X = X.values

    X = X.astype( float )

    X = normalize( X, norm_mean.values, 1 )
    log_hazard = pd.Series( np.dot( X, params_ ), index=index )
    return  log_hazard

def _calculate_variance_matrix(hessian,norm_std,zr):
    """

    :param hessian: second order derivative
    :param norm_std: standard deviation
    :return: variance matrix
    """

    if hessian.size > 0:
        variance_matrix_ = pd.DataFrame(
            -np.linalg.inv( hessian ) / np.outer( norm_std, norm_std ), index=zr.axes[0],
            columns=zr.axes[0]
        )
    else:
        variance_matrix_ = pd.DataFrame( index=zr.axes[0], columns=zr.axes[0] )
    return variance_matrix_

def _calculate_standard_errors(variance_matrix_,params_):
    """

    :return: Pandas Series of the standard errors

    This method calculates the standard errors out of the variance matrix.
    """
    se = np.sqrt(variance_matrix_.values.diagonal() )
    return pd.Series( se, name="se", index=params_.index )

def _calculate_z_values(params_,standard_errors_) -> pd.Series:
    """

    :return: z-values

    This method calculates the z-values.
    """
    return params_ / standard_errors_

def _calculate_p_values(params_,standard_errors_) -> np.ndarray:
    """

    :return: pvalues

    This method calculates the pvalues.
    """
    U = _calculate_z_values(params_,standard_errors_) ** 2
    return stats.chi2.sf( U, 1 )

def _calculate_confidence_intervals(alpha,params_,standard_errors_) -> pd.DataFrame:
    """

    :param alpha: normally 0.05
    :return: confidence intervals (the 95% lower and upper bound)
    """
    ci = 100 * (1 - alpha)
    z = inv_normal_cdf( 1 - alpha / 2 )
    se = standard_errors_
    hazards = params_.values
    return pd.DataFrame(
        np.c_[hazards - z * se, hazards + z * se],
        columns=["%g%% lower-bound" % ci, "%g%% upper-bound" % ci],
        index=params_.index,
    )