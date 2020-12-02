This is a federated Cox proportional hazard model app which is based on the fc federated flask image.
This app overwrites the api.py as well as the web.py.
In the templates directory are the html templates for this app.
In the static directory are the html images, css files and fonts required for the html templates.
In the requirements.txt are the project specific python requirements.
The build.sh script automatically builds the federated app with the correct name. (bash build.sh)
Before running the build.sh: first run the build.sh from the fc federated base, then from the fc federated flask and then the one from this app
To test run the federated Cox proportional hazard model app, the following steps need to be performed:

1. Run controller-frontend, controller and global-API.
2. Open frontend page: https://localhost:4200/app-test.
3. Determine the docker image name (app_name:version), the number of clients, and the directory containing the files for each simulated site.
4. Press start button to create multiple docker containers simulating the different sites and to test the federated Cox model.
