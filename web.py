import redis
import rq
from flask import Blueprint, current_app, render_template, request, send_file, Response, redirect,url_for, flash
from fc_app.api import redis_get, redis_set, preprocessing
import pandas as pd
from io import BytesIO
from matplotlib.figure import Figure
from lifelines.utils import inv_normal_cdf
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time

r = redis.Redis(host='localhost', port=6379, db=0)
tasks = rq.Queue('fc_tasks', connection=r)
web_bp = Blueprint('web', __name__)


@web_bp.route('/', methods=['GET'])
def root():
    """
    decides which HTML page content will be shown to the user
    :return: step == 'start': the setup is not yet finished and the program cannot start yet
             step == 'setup_master': master can define parameters and upload data
             step == 'setup_slave': client can upload data
             step == 'final': the results of the global mean will be shown
             else: the calculations are not yet finished
    """
    step = redis_get('step')

    if step == 'start':
        return render_template('setup.html')

    elif step == 'setup_master':
        current_app.logger.info('[WEB] Before rendering start_client.html')
        if redis_get('master'):
            return render_template('setup_master.html')
        else:
            return render_template('setup.html')

    elif step=='setup_slave':
        if not redis_get('master'):
            # get information from master to show in slaves setup html page
            max_steps = redis_get('max_steps')
            precision = redis_get('precision')
            duration_col = redis_get('duration_col')
            event_col = redis_get('event_col')
            covariates = redis_get('master_covariates')
            covariates = ', '.join(covariates)
            l1_ratio = redis_get('l1_ratio')
            penalization = redis_get('penalization')
            intersection = redis_get('intersection')


            if intersection==0:
                intersect = 'No. All covariates needs to be in the dataset.'
            else:
                intersect = 'Yes. The intersection of covariates of all clients is used.'
            if penalization==0:
                penalizer = 'no'
            else:
                if l1_ratio==0:
                    penalizer = 'lasso penalized regression with penalizer: '+str(penalization)
                elif l1_ratio==1:
                    penalizer = 'ridge penalized regression with penalizer: '+str(penalization)
                else:
                    penalizer = 'elastic net penalized regression with penalizer: '+str(penalization) + ' and l1_ratio: '+str(l1_ratio)

            return render_template('start_client.html',max_steps = max_steps, precision = precision, duration_col = duration_col, event_col = event_col, covariates = covariates, penalizer=penalizer,intersect=intersect)
        else:
            return render_template('calculations.html')

    elif step == 'final':
        result = redis_get( 'result' )
        penalization = redis_get( 'penalization' )
        # if a variable selection method was executed, just the covariates with p-values<=0.05 are shown in the final.html site
        if (penalization == 0):
            result_html = result.to_html( classes='data', header='true' )
        else:
            result = result[ result[ 'p' ] <= 0.05 ]
            result_html = result.to_html( classes='data', header='true' )

        c_index = redis_get( 'c_index' )

        intersection = redis_get( 'intersection' )

        if intersection == 1:
            if redis_get( 'master' ):
                cov = redis_get( 'covariates' )
            else:
                cov = redis_get( 'master_covariates' )

            intersect_covariates = redis_get( 'intersected_covariates' )
            # find those covariates who fall away
            fall_away = set( cov ) ^ set( intersect_covariates )
            current_app.logger.info( f'[WEB] fall_away covariates: {fall_away}' )
            if fall_away:
                covariates = ", ".join( str( cov ) for cov in fall_away )
                warning = "Some clients didn't have all covariates. The missing ones are :" + covariates
                flash( warning )
        return render_template( 'final.html', tables=[ result_html ], c_index=c_index )
    else:
        return render_template('calculations.html')

@web_bp.route('/params', methods=['GET'])
def params():
    """
    :return: current parameter values as a HTML page
    """
    master = redis_get('master')
    step = redis_get('step')
    local_norm = redis_get('local_norm')
    global_norm = redis_get('global_norm')
    data = redis_get('data')
    available = redis_get('available')
    return f"""
        master: {master}
        step: {step}
        local normalization: {local_norm}
        global normalization: {global_norm}
        data: {data}
        available: {available}
        """

@web_bp.route('/run', methods=['POST'])
def run():
    """
    POST request to /run with the data as a file
    step == 'setup_master': the file of the master will be read and some parameters will be saved
    step == 'setup_client': the file of the clients will be read
    step == 'final': calculations are done, a GET request to '/' will be send
    else: a message for the relevant step will be shown
    :return: HTML page with content to the relevant step
    """
    cur_step = redis_get('step')

    if cur_step == 'start':
        current_app.logger.info('[WEB] POST request to /run in step "start" -> wait setup not finished')
        return 'Wait until setup is done'

    elif cur_step == 'setup_master':
        if redis_get('master'):
            result = request.form
            # check if a file was uploaded
            if 'file' in request.files :
                file = request.files['file']
                if file.filename != '':
                    file_type = result[ 'file_type' ]
                    if file_type=='csv':
                        try:
                            data = pd.read_csv(file,sep=',',encoding="latin-1")
                        except Exception:
                            error_message = 'Upload file not in selected format!'
                            flash( error_message )
                            return request.referrer

                    elif file_type=='tsv':
                        try:
                            data = pd.read_csv( file, sep='\t', encoding="latin-1" )
                        except Exception:
                            error_message = 'Upload file not in selected format!'
                            flash( error_message )
                            return request.referrer

                    redis_set('data', data)
                    current_app.logger.info('[WEB] File successfully uploaded and processed')

                    duration_col = result['duration_col']
                    event_col = result['event_col']

                    current_app.logger.info( f'[WEB] Duration and event column selected' )

                    max_steps = result['max_steps']
                    precision = result['precision']
                    redis_set('max_steps',int(max_steps))
                    redis_set('precision',float(precision))

                    intersection = result['intersection']
                    if intersection=='intersect_yes':
                        #take the intersection of all covariates
                        redis_set('intersection',1)
                    elif intersection=='intersect_no':
                        #force clients to have all the same covariates
                        redis_set('intersection',0)

                    high_dimensional = result['hg_radio']
                    if high_dimensional=='no':
                        l1_ratio = 0.0
                        penalization=0.0
                    elif high_dimensional=='yes':
                        penalty = result['penalty']
                        if penalty=='lasso':
                            l1_ratio=0.0
                        elif penalty=='ridge':
                            l1_ratio=1.0
                        elif penalty=='elastic_net':
                            l1_ratio=result['l1_ratio']
                        penalization = result['penalization']

                    redis_set('penalization',float(penalization))
                    redis_set('l1_ratio',float(l1_ratio))


                    #check if event and duration column were specified
                    if (duration_col == '' or event_col == ''):
                        error_message = 'No duration or event column specified, please do setup again!'
                        flash(error_message)
                        return request.referrer
                    else:
                        #check if duration and event column are really columns of the data
                        if (duration_col in data.columns) and (event_col in data.columns):
                            #current_app.logger.info( f'[WEB] duration_col: {duration_col}' )
                            redis_set( 'duration_col', duration_col )
                            #current_app.logger.info( f'[WEB] event_col: {event_col}' )
                            redis_set( 'event_col', event_col )
                            #current_app.logger.info( '[WEB] Duration and event columns successfully uploaded' )

                            # the parameter will be send to the slaves
                            redis_set( 'available', True )
                            redis_set( 'step', 'preprocessing' )
                            preprocessing()
                            return render_template('calculations.html')

                        # if they do not exist in data
                        else:
                            current_app.logger.info( '[WEB] Event and/or duration column not in dataset' )
                            error_message = 'Duration and/or event column not in dataset, please do setup again!'
                            flash( error_message )
                            return request.referrer
                else:
                    # if no file was uploaded, show again setup page with error message
                    current_app.logger.info( '[WEB] No File was uploaded' )
                    error_message = 'No file was uploaded, please do setup again!'
                    flash( error_message )
                    return redirect(url_for('web.root'))

            else:
                # if no file was uploaded, show again setup page with error message
                current_app.logger.info('[WEB] No File was uploaded')
                error_message = 'No file was uploaded, please do setup again!'
                flash( error_message )
                return redirect( url_for( 'web.root' ) )

    elif cur_step == 'setup_slave':
        if not redis_get('master'):
            result = request.form
            if 'file' in request.files:
                file = request.files[ 'file' ]
                if file.filename !='':
                    file_type = result[ 'file_type' ]

                    if file_type == 'csv':
                        try:
                            data = pd.read_csv( file, sep=',', encoding="latin-1" )
                        except Exception:
                            error_message = 'Upload file not in selected format!'
                            flash( error_message )
                            return request.referrer
                    elif file_type == 'tsv':
                        try:
                            data = pd.read_csv( file, sep='\t', encoding="latin-1" )
                        except Exception:
                            error_message = 'Upload file not in selected format!'
                            flash( error_message )
                            return request.referrer



                    redis_set( 'data', data )
                    current_app.logger.info( '[WEB] File successfully uploaded and processed' )


                    master_covariates = redis_get('master_covariates')
                    duration_col = redis_get('duration_col')
                    event_col = redis_get('event_col')
                    intersection = redis_get('intersection')

                    # check if duration col and event col are in the dataset
                    if (duration_col in data.columns) and (event_col in data.columns):
                        local_covariates = data.columns.values
                        index = np.argwhere( local_covariates == duration_col )
                        local_covariates = np.delete( local_covariates, index )
                        index = np.argwhere( local_covariates == event_col )
                        local_covariates = np.delete( local_covariates, index )

                        if (intersection==1):
                            # intersection of covariates is allowed, check if intersection is not null
                            intersect_covariates = pd.Series( np.intersect1d( local_covariates, master_covariates ) )
                            if not intersect_covariates.empty:
                                redis_set( 'master_step', 'master_intersect' )
                                redis_set( 'slave_step', 'slave_intersect' )

                                redis_set( 'step', 'preprocessing' )
                                preprocessing()

                                return render_template( 'calculations.html' )
                            else:
                                # the intersection of master and client is null, show again setup page with error message
                                current_app.logger.info( '[WEB] Intersection of covariates is null' )
                                error_message = 'Not a single covariant matches the specified , please do setup again!'
                                flash( error_message )
                                return redirect( url_for( 'web.root' ) )


                        elif (intersection==0):
                            # check if all covariates are exactly the same as those of the master
                            if (set(master_covariates)==set(local_covariates)):
                                redis_set( 'master_step', 'master_intersect' )
                                redis_set( 'slave_step', 'slave_intersect' )

                                redis_set( 'step', 'preprocessing' )
                                preprocessing()

                                return render_template( 'calculations.html' )
                            else:
                                # the covariates need to be all in the data of the slave
                                current_app.logger.info( '[WEB] No intersection provided, covariates differ from those of master' )
                                error_message = 'All given covariates must be in the dataset. Intersection not provided, please do setup again!'
                                flash( error_message )
                                return redirect( url_for( 'web.root' ) )

                    else:
                        # duration and event col are not in the dataset, show again setup page with error message
                        current_app.logger.info( '[WEB] Duration and/or event Column not in dataset' )
                        error_message = 'Duration and/or event column not in dataset, please do setup again!'
                        flash( error_message )
                        return redirect( url_for( 'web.root' ) )


                else:
                    # if no file was uploaded, show again setup page with error message
                    current_app.logger.info( '[WEB] No File was uploaded' )
                    error_message = 'No file was uploaded, please do setup again!'
                    flash( error_message )
                    return redirect( url_for( 'web.root' ) )


            else:
                # if no file was uploaded, show again setup page with error message
                current_app.logger.info( '[WEB] No File was uploaded' )
                error_message = 'No file was uploaded, please do setup again!'
                flash( error_message )
                return redirect( url_for( 'web.root' ) )

    elif cur_step == 'final':
        current_app.logger.info('[WEB] POST request to /run in step "final" -> GET request to "/"')
        # TODO weiterleitung zu route /

    else:
        current_app.logger.info(f'[WEB] POST request to /run in step "{cur_step}" -> wait calculations not finished')
        return render_template('calculations.html')

@web_bp.route('/download_results')
def download_results():
    result = redis_get('result')
    file_data = result.to_csv(sep='\t').encode()
    return send_file(BytesIO(file_data), attachment_filename='results.tsv', as_attachment=True)

@web_bp.route('/plot.png')
def download_plot():
    result = redis_get('result')
    params_ = result[ 'coef' ]
    standard_errors_ = result[ 'se(coef)' ]
    fig = create_figure(params_,standard_errors_)
    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(),mimetype='image/png')

def create_figure(params_,standard_errors_):
    fig = Figure()
    axis = fig.add_subplot(1,1,1)
    axis = plot(params_, standard_errors_, ax=axis )
    fig.tight_layout()
    return fig

def plot(params_, standard_errors_, ax=None, **errorbar_kwargs):
    """
    Produces a visual representation of the coefficients (i.e. log hazard ratios), including their standard errors and magnitudes.

    Parameters
    ----------
    columns : list, optional
        specify a subset of the columns to plot
    hazard_ratios: bool, optional
        by default, ``plot`` will present the log-hazard ratios (the coefficients). However, by turning this flag to True, the hazard ratios are presented instead.
    errorbar_kwargs:
        pass in additional plotting commands to matplotlib errorbar command

    Returns
    -------
    ax: matplotlib axis
        the matplotlib axis that be edited.

    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()

    alpha = 0.05

    errorbar_kwargs.setdefault( "c", "k" )
    errorbar_kwargs.setdefault( "fmt", "s" )
    errorbar_kwargs.setdefault( "markerfacecolor", "white" )
    errorbar_kwargs.setdefault( "markeredgewidth", 1.25 )
    errorbar_kwargs.setdefault( "elinewidth", 1.25 )
    errorbar_kwargs.setdefault( "capsize", 3 )

    z = inv_normal_cdf( 1 - alpha / 2 )
    user_supplied_columns = True


    user_supplied_columns = False
    columns = params_.index

    yaxis_locations = list( range( len( columns ) ) )
    log_hazards = params_.loc[ columns ].values.copy()

    order = list( range( len( columns ) - 1, -1, -1 ) ) if user_supplied_columns else np.argsort( log_hazards )


    symmetric_errors = z * standard_errors_[ columns ].values
    ax.errorbar( log_hazards[ order ], yaxis_locations, xerr=symmetric_errors[ order ], **errorbar_kwargs )
    ax.set_xlabel( "log(HR) (%g%% CI)" % ((1 - alpha) * 100) )

    best_ylim = ax.get_ylim()
    ax.vlines(0, -2, len( columns ) + 1, linestyles="dashed", linewidths=1, alpha=0.65 )
    ax.set_ylim( best_ylim )

    tick_labels = [ columns[ i ] for i in order ]

    ax.set_yticks( yaxis_locations )
    ax.set_yticklabels( tick_labels )
    ax.set_title("Visual representation of the coefficients of the federated model")

    return ax