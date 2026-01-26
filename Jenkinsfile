pipeline {

    agent {
        label 'pytest'  // Etiqueta del agente
    }

    environment {
        VENV_DIR = 'venv'  // Nombre del entorno virtual
    }

    stages {
        stage('Preparar Entorno') {
            steps {
                // Crear entorno virtual
                sh 'python3 -m venv ${VENV_DIR}'
                // Activar el entorno virtual e instalar dependencias
                sh '''
                   . ${VENV_DIR}/bin/activate
                   pip install --upgrade pip
                   pip install -r requirements.txt
                '''
            }
        }
        stage('Ejecutar Tests') {
            steps {
                 //Instalar pytest y httpx
                sh '''
                    . ${VENV_DIR}/bin/activate
                    pip install pytest
                    pip install httpx
                '''
                // Activar el entorno virtual y ejecutar pytest
                sh '''
                   . ${VENV_DIR}/bin/activate
                   pytest --disable-warnings --junitxml=report.xml
                '''
            }
        }
    }

    post {
        always {
            // Generación de reportes 
            junit 'report.xml'
            echo 'Job finalizado.'
        }
    }
}
