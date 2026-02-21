pipeline {
    agent { label 'agent-45' }

    options {
        buildDiscarder(logRotator(numToKeepStr: '5'))
        timestamps()
        timeout(time: 60, unit: 'MINUTES')
    }

    environment {
        REGISTRY   = "192.168.1.86:5000"
        IMAGE_NAME = "hopfieldspp"
        NO_PROXY   = 'localhost,127.0.0.1,192.168.1.0/24,192.168.1.86,192.168.1.62,192.168.1.45'
        no_proxy   = 'localhost,127.0.0.1,192.168.1.0/24,192.168.1.86,192.168.1.62,192.168.1.45'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Build Image') {
            steps {
                echo 'Building Docker image...'
                sh "docker build -t ${REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER} -t ${REGISTRY}/${IMAGE_NAME}:latest ."
            }
        }

        stage('Code Quality Checks') {
            parallel {
                stage('Lint') {
                    steps {
                        catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                            sh """
                            docker run --rm --user root \
                                ${REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER} \
                                sh -c 'pip install --quiet flake8 && flake8 src/ --max-line-length=120 --count --statistics'
                            """
                        }
                    }
                }

                stage('Security Checks') {
                    steps {
                        catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                            echo 'Checking for security vulnerabilities...'
                            sh """
                            docker run --rm --user root \
                                ${REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER} \
                                sh -c 'pip install --quiet pip-audit && pip-audit -r requirements.txt'
                            """
                        }
                    }
                }
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running test suite with coverage...'
                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                    script {
                        try {
                            sh '''
                            docker run --name test-spp-$BUILD_NUMBER \
                                --user root \
                                -v "$WORKSPACE/tests:/app/tests" \
                                $REGISTRY/$IMAGE_NAME:$BUILD_NUMBER \
                                sh -c "pip install --quiet pytest httpx pytest-cov scipy && python -m pytest tests/ -v \
                                    --junitxml=test-results.xml \
                                    --cov=src \
                                    --cov-report=xml:coverage.xml \
                                    --cov-report=term-missing \
                                    --disable-warnings"
                            '''
                        } finally {
                            sh "docker cp test-spp-${env.BUILD_NUMBER}:/app/test-results.xml ${env.WORKSPACE}/test-results.xml || true"
                            sh "docker cp test-spp-${env.BUILD_NUMBER}:/app/coverage.xml ${env.WORKSPACE}/coverage.xml || true"
                            sh "docker rm test-spp-${env.BUILD_NUMBER} || true"
                        }
                    }
                }
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                    archiveArtifacts artifacts: 'coverage.xml', allowEmptyArchive: true, fingerprint: true
                }
            }
        }

        stage('SonarQube Analysis') {
            steps {
                withCredentials([usernamePassword(
                    credentialsId: 'sonarqube-credentials',
                    usernameVariable: 'SONAR_USER',
                    passwordVariable: 'SONAR_PASS'
                )]) {
                    sh """
                        docker run --rm \
                            -e SONAR_USER="\$SONAR_USER" \
                            -e SONAR_PASS="\$SONAR_PASS" \
                            -v "${env.WORKSPACE}:/usr/src" \
                            sonarsource/sonar-scanner-cli \
                            -Dsonar.projectKey=HopfieldSPP \
                            -Dsonar.sources=src \
                            -Dsonar.tests=tests \
                            -Dsonar.python.version=3.9 \
                            -Dsonar.python.coverage.reportPaths=coverage.xml \
                            -Dsonar.host.url=http://192.168.1.86:9000 \
                            -Dsonar.login="\$SONAR_USER" \
                            -Dsonar.password="\$SONAR_PASS" \
                            -Dsonar.scm.disabled=true
                    """
                }
            }
        }

        stage('Push to Registry') {
            steps {
                echo "Pushing image to ${REGISTRY}..."
                sh "docker push ${REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER}"
                sh "docker push ${REGISTRY}/${IMAGE_NAME}:latest"
            }
        }
    }

    post {
        always {
            sh 'rm -f test-results.xml coverage.xml || true'
            sh "docker rmi ${REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER} || true"
            cleanWs()
        }
        success {
            echo 'Pipeline succeeded!'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}
