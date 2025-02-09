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
                //Instalar pytest
                sh '''
                    . ${VENV_DIR}/bin/activate
                    pip install pytest
                '''
            }
            steps {
                // Activar el entorno virtual y ejecutar pytest
                sh '''
                   . ${VENV_DIR}/bin/activate
                   pytest --maxfail=1 --disable-warnings -q
                '''
            }
        }
    }

    post {
        always {
            // Limpieza o generación de reportes (opcional)
            echo 'Job finalizado.'
        }
    }
}
