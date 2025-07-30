# ===== CLOUD FOUNDRY BUILDPACK BEHAVIOR =====

“””
Cloud Foundry Python Buildpack Dependency Resolution Order:

1. Checks for requirements.txt in root → pip install -r requirements.txt
1. If no requirements.txt, checks setup.py → pip install .
1. If both exist, requirements.txt takes PRECEDENCE

This means: Keep setup.py minimal, use requirements.txt for all dependencies!
“””

# ===== CORRECT APPROACH: Minimal setup.py =====

# setup.py (Keep this MINIMAL)

from setuptools import setup, find_packages

setup(
name=‘my-cf-app’,
version=‘1.0.0’,
description=‘Cloud Foundry Python Application’,
author=‘Your Name’,
author_email=‘your.email@company.com’,
packages=find_packages(),
python_requires=’>=3.11’,

```
# ✅ MINIMAL - Only core runtime dependencies that define your package
install_requires=[
    'Flask>=2.3.3',         # Core web framework
    'gunicorn>=21.2.0',     # WSGI server
],

# ✅ Include package data and config files
include_package_data=True,
zip_safe=False,

# Package metadata
classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3.11',
],
```
)

# ===== COMPREHENSIVE requirements.txt =====

# requirements.txt (This is where ALL your 100+ packages go)

“””

# Web Framework and Server

Flask==2.3.3
Werkzeug==2.3.7
gunicorn==21.2.0

# Database

SQLAlchemy==2.0.23
psycopg2-binary==2.9.7
alembic==1.12.1

# Authentication & Security

Flask-Login==0.6.3
Flask-JWT-Extended==4.5.3
bcrypt==4.0.1
cryptography==41.0.7

# API and Serialization

marshmallow==3.20.1
Flask-RESTful==0.3.10
requests==2.31.0
urllib3==2.0.7

# Caching and Redis

redis==5.0.1
Flask-Caching==2.1.0

# Monitoring and Logging

prometheus-client==0.18.0
structlog==23.1.0
sentry-sdk==1.36.0

# Data Processing

pandas==2.1.3
numpy==1.25.2
scipy==1.11.4

# AWS Services

boto3==1.29.7
botocore==1.32.7

# Configuration

python-decouple==3.8
PyYAML==6.0.1

# Testing (only if needed in runtime)

pytest==7.4.3
pytest-cov==4.1.0

# … (90+ more packages)

“””

# ===== WHY THIS APPROACH WORKS =====

“””
Cloud Foundry Python Buildpack Process:

1. Upload & Extract:
- cf push uploads your zip file
- Buildpack extracts to /tmp/staged/app/
1. Dependency Detection:
   /tmp/staged/app/
   ├── my_app/
   ├── setup.py          ← Buildpack sees this
   ├── requirements.txt   ← Buildpack prioritizes this!
   ├── Procfile
   └── .pip/pip.conf     ← Buildpack uses this for repo config
1. Dependency Installation:
- Buildpack finds requirements.txt
- Runs: pip install -r requirements.txt
- Uses your .pip/pip.conf for Artifactory
- Installs all 100+ packages from your internal repo
1. Application Start:
- Uses Procfile to start your app
- All dependencies are available
  “””

# ===== PROJECT STRUCTURE FOR CF =====

“””
my_cf_project/
├── my_app/
│   ├── **init**.py
│   ├── app.py
│   ├── models/
│   ├── services/
│   └── utils/
├── tests/
├── .pip/
│   └── pip.conf          # Artifactory configuration
├── .cfignore
├── MANIFEST.in           # Includes requirements.txt
├── Procfile              # gunicorn command
├── manifest.yml          # CF deployment config
├── requirements.txt      # ✅ ALL 100+ packages here
├── runtime.txt           # Python version
└── setup.py              # ✅ MINIMAL package definition
“””

# ===== MANIFEST.in (Updated) =====

“””

# MANIFEST.in - Ensure requirements.txt is included in zip

include Procfile
include requirements.txt     # ✅ Critical for CF buildpack
include runtime.txt
include manifest.yml
recursive-include .pip *
include .cfignore
recursive-include my_app *.py
recursive-include tests *.py
“””

# ===== BUILDPACK DETECTION LOGIC =====

“””
Python Buildpack Decision Tree:

1. requirements.txt exists in root?
   → YES: pip install -r requirements.txt (✅ Your case)
   → NO: Go to step 2
1. setup.py exists?
   → YES: pip install . (uses install_requires)
   → NO: Skip dependency installation
1. Both exist?
   → requirements.txt takes priority
   → setup.py install_requires is IGNORED
   “””

# ===== VERIFICATION SCRIPT =====

“””
#!/bin/bash

# verify_package.sh - Test your packaging approach

echo “🔍 Verifying Cloud Foundry package structure…”

# Create the package

python setup.py sdist –formats=zip

# Extract and examine

TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR
unzip -q ../dist/my-cf-app-1.0.0.zip
cd my-cf-app-1.0.0

echo “📦 Package contents:”
ls -la

echo “📋 Checking critical files:”
if [ -f “requirements.txt” ]; then
echo “✅ requirements.txt found”
echo “   Packages count: $(wc -l < requirements.txt)”
else
echo “❌ requirements.txt MISSING!”
fi

if [ -f “setup.py” ]; then
echo “✅ setup.py found”
echo “   install_requires: $(grep -A 10 ‘install_requires’ setup.py | wc -l) lines”
else
echo “❌ setup.py MISSING!”
fi

if [ -d “.pip” ] && [ -f “.pip/pip.conf” ]; then
echo “✅ .pip/pip.conf found (Artifactory config)”
else
echo “❌ .pip/pip.conf MISSING!”
fi

if [ -f “Procfile” ]; then
echo “✅ Procfile found”
cat Procfile
else
echo “❌ Procfile MISSING!”
fi

# Cleanup

cd - && rm -rf $TEMP_DIR

echo “🎯 Result: CF buildpack will use requirements.txt for dependencies!”
“””

# ===== COMMON ANTI-PATTERNS TO AVOID =====

“””
❌ DON’T DO THIS - Duplicating packages in setup.py:

setup(
install_requires=[
‘Flask==2.3.3’,
‘SQLAlchemy==2.0.23’,
‘pandas==2.1.3’,
‘numpy==1.25.2’,
‘boto3==1.29.7’,
# … copying all 100+ packages
]
)

Problems:

- Duplication and maintenance nightmare
- Version conflicts between setup.py and requirements.txt
- CF buildpack ignores install_requires when requirements.txt exists
- Violates separation of concerns

✅ DO THIS - Minimal setup.py + comprehensive requirements.txt:

# setup.py

install_requires=[
‘Flask>=2.3.3’,      # Only core framework
‘gunicorn>=21.2.0’,  # Only WSGI server
]

# requirements.txt (all 100+ packages with exact versions)

Flask==2.3.3
gunicorn==21.2.0
SQLAlchemy==2.0.23
pandas==2.1.3

# … all other packages

“””

# ===== DEPENDENCY MANAGEMENT BEST PRACTICES =====

“””
For Large Applications with 100+ Dependencies:

1. Use requirements.txt for ALL runtime dependencies
1. Consider requirements-dev.txt for development dependencies
1. Use pip-tools for dependency management:
- requirements.in (high-level dependencies)
- requirements.txt (pinned versions from pip-compile)
1. Example structure:
   requirements/
   ├── base.txt           # Core dependencies
   ├── production.txt     # Prod-specific (monitoring, etc.)
   ├── development.txt    # Dev tools (testing, linting)
   └── requirements.txt   # Combined for CF deployment
   “””

# ===== CICD BUILD VERIFICATION =====

“””

# In your CI/CD pipeline, add verification:

- name: Verify Package Structure
  script:
  - python setup.py sdist –formats=zip
  - unzip -t dist/my-cf-app-1.0.0.zip
  - unzip -l dist/my-cf-app-1.0.0.zip | grep requirements.txt
  - echo “✅ requirements.txt included in package”
  - unzip -l dist/my-cf-app-1.0.0.zip | grep “.pip/pip.conf”
  - echo “✅ pip.conf included in package”
    “””

# ===== FINAL RECOMMENDATION =====

“””
For your 100+ package Cloud Foundry application:

✅ Keep setup.py minimal (only Flask + gunicorn)
✅ Put ALL 100+ packages in requirements.txt  
✅ CF buildpack will automatically use requirements.txt
✅ Your Artifactory configuration will work perfectly
✅ No duplication or maintenance issues

This is the standard, scalable approach used by enterprise CF deployments.
“””


**No, you should NOT include all 100+ packages in `install_requires` in setup.py!**

You’re absolutely right - the Cloud Foundry Python buildpack will automatically pick up and install from `requirements.txt` when it’s in the root of your zip file. This is the standard and recommended approach.

Here’s why and how it works:​​​​​​​​​​​​​​​​

## Summary:

**✅ Correct Approach:**

- **setup.py**: Keep minimal (only 2-3 core packages like Flask, gunicorn)
- **requirements.txt**: All 100+ packages with exact versions
- **CF Buildpack**: Automatically finds and uses `requirements.txt`

**Why this works:**

1. **CF Python buildpack priority**: `requirements.txt` > `setup.py`
1. **Separation of concerns**: setup.py defines the package, requirements.txt defines runtime environment
1. **No duplication**: Maintain dependencies in one place
1. **Buildpack automatically runs**: `pip install -r requirements.txt` using your Artifactory config

**Key verification:**

```bash
# After running: python setup.py sdist --formats=zip
unzip -l dist/my-cf-app-1.0.0.zip | grep requirements.txt
# Should show: requirements.txt in the zip root
```

The buildpack will find `requirements.txt` in your zip file root and install all 100+ packages from your internal Artifactory repository. This is exactly how it’s designed to work!​​​​​​​​​​​​​​​​

# ===== cicd.yml (GitLab CI/CD) =====

# Place this file in the root of your project

stages:

- validate
- build
- test
- package
- deploy

variables:
PYTHON_VERSION: “3.11”
PIP_CACHE_DIR: “$CI_PROJECT_DIR/.cache/pip”
APP_NAME: “my-cf-app”
PACKAGE_VERSION: “1.0.0”

# Cache pip dependencies

cache:
paths:
- .cache/pip
- venv/

# ===== VALIDATION STAGE =====

validate:
stage: validate
image: python:${PYTHON_VERSION}
script:
- echo “Validating project structure…”
- ls -la
- test -f setup.py || (echo “setup.py not found!” && exit 1)
- test -f requirements.txt || (echo “requirements.txt not found!” && exit 1)
- test -f Procfile || (echo “Procfile not found!” && exit 1)
- test -f MANIFEST.in || (echo “MANIFEST.in not found!” && exit 1)
- test -d .pip || (echo “.pip directory not found!” && exit 1)
- test -f .pip/pip.conf || (echo “pip.conf not found!” && exit 1)
- echo “✅ All required files present”
only:
- merge_requests
- main
- develop

# ===== BUILD STAGE =====

build:
stage: build
image: python:${PYTHON_VERSION}
before_script:
- python -m venv venv
- source venv/bin/activate
- pip install –upgrade pip setuptools wheel
# Configure pip to use internal Artifactory
- mkdir -p ~/.pip
- cp .pip/pip.conf ~/.pip/pip.conf
# Set Artifactory credentials from CI variables
- export PIP_INDEX_URL=“https://${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}@${ARTIFACTORY_HOST}/api/pypi/pypi-virtual/simple/”
- export PIP_TRUSTED_HOST=”${ARTIFACTORY_HOST}”
script:
- echo “Installing dependencies…”
- pip install -r requirements.txt
- echo “✅ Build completed successfully”
artifacts:
paths:
- venv/
expire_in: 1 hour
only:
- merge_requests
- main
- develop

# ===== TEST STAGE =====

test:
stage: test
image: python:${PYTHON_VERSION}
dependencies:
- build
before_script:
- source venv/bin/activate
script:
- echo “Running tests…”
- python -m pytest tests/ -v –junitxml=report.xml –cov=my_app –cov-report=xml
- echo “Running linting…”
- flake8 my_app/
- echo “✅ Tests passed”
artifacts:
reports:
junit: report.xml
coverage_report:
coverage_format: cobertura
path: coverage.xml
expire_in: 1 week
coverage: ‘/TOTAL.*\s+(\d+%)$/’
only:
- merge_requests
- main
- develop

# ===== PACKAGE STAGE (CRITICAL FOR CF DEPLOYMENT) =====

package:
stage: package
image: python:${PYTHON_VERSION}
dependencies:
- build
- test
before_script:
- source venv/bin/activate
- pip install –upgrade setuptools wheel
script:
- echo “Creating source distribution package…”
# This is the KEY command for Cloud Foundry deployment
- python setup.py sdist –formats=zip
- echo “Package created successfully:”
- ls -la dist/
- echo “Package contents:”
- unzip -l dist/${APP_NAME}-${PACKAGE_VERSION}.zip | head -20
- echo “✅ Package ready for deployment”
artifacts:
name: “${APP_NAME}-${CI_COMMIT_SHA:0:8}”
paths:
- dist/${APP_NAME}-${PACKAGE_VERSION}.zip
expire_in: 1 week
reports:
# Create deployment report
- deployment_info.json
after_script:
# Generate deployment metadata
- |
cat > deployment_info.json << EOF
{
“app_name”: “${APP_NAME}”,
“version”: “${PACKAGE_VERSION}”,
“commit_sha”: “${CI_COMMIT_SHA}”,
“pipeline_id”: “${CI_PIPELINE_ID}”,
“package_file”: “dist/${APP_NAME}-${PACKAGE_VERSION}.zip”,
“created_at”: “$(date -Iseconds)”
}
EOF
only:
- main
- develop

# ===== DEPLOY STAGE =====

deploy_staging:
stage: deploy
image:
name: cloudfoundry/cf-cli-resource:latest
entrypoint: [””]
dependencies:
- package
variables:
CF_API: “${CF_API_STAGING}”
CF_ORG: “${CF_ORG_STAGING}”
CF_SPACE: “${CF_SPACE_STAGING}”
CF_APP_NAME: “${APP_NAME}-staging”
before_script:
- cf version
- cf api ${CF_API}
- cf auth ${CF_USERNAME} ${CF_PASSWORD}
- cf target -o ${CF_ORG} -s ${CF_SPACE}
script:
- echo “Deploying to staging environment…”
# Set environment variables for Artifactory
- cf set-env ${CF_APP_NAME} PIP_INDEX_URL “https://${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}@${ARTIFACTORY_HOST}/api/pypi/pypi-virtual/simple/”
- cf set-env ${CF_APP_NAME} PIP_TRUSTED_HOST “${ARTIFACTORY_HOST}”
# Deploy using the zip package
- cf push ${CF_APP_NAME} -p dist/${APP_NAME}-${PACKAGE_VERSION}.zip
- echo “✅ Deployed to staging successfully”
- cf app ${CF_APP_NAME}
environment:
name: staging
url: https://${CF_APP_NAME}.${CF_APPS_DOMAIN}
only:
- develop

deploy_production:
stage: deploy
image:
name: cloudfoundry/cf-cli-resource:latest
entrypoint: [””]
dependencies:
- package
variables:
CF_API: “${CF_API_PROD}”
CF_ORG: “${CF_ORG_PROD}”
CF_SPACE: “${CF_SPACE_PROD}”
CF_APP_NAME: “${APP_NAME}”
before_script:
- cf version
- cf api ${CF_API}
- cf auth ${CF_USERNAME} ${CF_PASSWORD}
- cf target -o ${CF_ORG} -s ${CF_SPACE}
script:
- echo “Deploying to production environment…”
# Set environment variables for Artifactory
- cf set-env ${CF_APP_NAME} PIP_INDEX_URL “https://${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}@${ARTIFACTORY_HOST}/api/pypi/pypi-virtual/simple/”
- cf set-env ${CF_APP_NAME} PIP_TRUSTED_HOST “${ARTIFACTORY_HOST}”
# Deploy using the zip package
- cf push ${CF_APP_NAME} -p dist/${APP_NAME}-${PACKAGE_VERSION}.zip
- echo “✅ Deployed to production successfully”
- cf app ${CF_APP_NAME}
environment:
name: production
url: https://${CF_APP_NAME}.${CF_APPS_DOMAIN}
when: manual
only:
- main

# ===== Jenkinsfile =====

# Place this file in the root of your project as “Jenkinsfile”

# Jenkinsfile (Declarative Pipeline)

pipeline {
agent any

```
parameters {
    choice(
        name: 'ENVIRONMENT',
        choices: ['staging', 'production'],
        description: 'Target deployment environment'
    )
    booleanParam(
        name: 'SKIP_TESTS',
        defaultValue: false,
        description: 'Skip test execution'
    )
}

environment {
    PYTHON_VERSION = '3.11'
    APP_NAME = 'my-cf-app'
    PACKAGE_VERSION = '1.0.0'
    PIP_CACHE_DIR = "${WORKSPACE}/.cache/pip"
    
    // Artifactory credentials (stored in Jenkins credentials)
    ARTIFACTORY_CREDS = credentials('artifactory-credentials')
    ARTIFACTORY_HOST = 'your-artifactory.company.com'
    
    // Cloud Foundry credentials
    CF_CREDS = credentials('cf-credentials')
}

stages {
    stage('Checkout') {
        steps {
            echo '🔄 Checking out source code...'
            checkout scm
            sh 'ls -la'
        }
    }
    
    stage('Validate') {
        steps {
            echo '✅ Validating project structure...'
            script {
                def requiredFiles = ['setup.py', 'requirements.txt', 'Procfile', 'MANIFEST.in', '.pip/pip.conf']
                requiredFiles.each { file ->
                    if (!fileExists(file)) {
                        error("Required file ${file} not found!")
                    }
                }
            }
            echo '✅ All required files present'
        }
    }
    
    stage('Setup Python Environment') {
        steps {
            echo '🐍 Setting up Python environment...'
            sh '''
                python${PYTHON_VERSION} -m venv venv
                source venv/bin/activate
                pip install --upgrade pip setuptools wheel
                
                # Configure pip for Artifactory
                mkdir -p ~/.pip
                cp .pip/pip.conf ~/.pip/pip.conf
                
                echo "✅ Python environment ready"
            '''
        }
    }
    
    stage('Install Dependencies') {
        steps {
            echo '📦 Installing dependencies...'
            sh '''
                source venv/bin/activate
                
                # Set Artifactory credentials
                export PIP_INDEX_URL="https://${ARTIFACTORY_CREDS_USR}:${ARTIFACTORY_CREDS_PSW}@${ARTIFACTORY_HOST}/api/pypi/pypi-virtual/simple/"
                export PIP_TRUSTED_HOST="${ARTIFACTORY_HOST}"
                
                # Install dependencies
                pip install -r requirements.txt
                pip install pytest flake8 coverage pytest-cov
                
                echo "✅ Dependencies installed"
            '''
        }
    }
    
    stage('Run Tests') {
        when {
            not { params.SKIP_TESTS }
        }
        steps {
            echo '🧪 Running tests...'
            sh '''
                source venv/bin/activate
                
                # Run tests with coverage
                python -m pytest tests/ -v --junitxml=test-results.xml --cov=my_app --cov-report=xml
                
                # Run linting
                flake8 my_app/ || echo "Linting warnings found"
                
                echo "✅ Tests completed"
            '''
        }
        post {
            always {
                // Publish test results
                junit 'test-results.xml'
                // Publish coverage report
                publishCoverage adapters: [coberturaAdapter('coverage.xml')], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
            }
        }
    }
    
    stage('Create Package') {
        steps {
            echo '📦 Creating deployment package...'
            sh '''
                source venv/bin/activate
                
                # This is the CRITICAL command for Cloud Foundry
                echo "Creating source distribution..."
                python setup.py sdist --formats=zip
                
                # Verify package creation
                echo "Package created:"
                ls -la dist/
                
                echo "Package contents:"
                unzip -l dist/${APP_NAME}-${PACKAGE_VERSION}.zip | head -20
                
                # Create deployment metadata
                cat > deployment_info.json << EOF
```

{
“app_name”: “${APP_NAME}”,
“version”: “${PACKAGE_VERSION}”,
“build_number”: “${BUILD_NUMBER}”,
“git_commit”: “${GIT_COMMIT}”,
“package_file”: “dist/${APP_NAME}-${PACKAGE_VERSION}.zip”,
“created_at”: “$(date -Iseconds)”
}
EOF

```
                echo "✅ Package ready for deployment"
            '''
        }
        post {
            success {
                // Archive the deployment package
                archiveArtifacts artifacts: "dist/${APP_NAME}-${PACKAGE_VERSION}.zip", fingerprint: true
                archiveArtifacts artifacts: 'deployment_info.json', fingerprint: true
            }
        }
    }
    
    stage('Deploy to Cloud Foundry') {
        steps {
            echo "🚀 Deploying to ${params.ENVIRONMENT}..."
            script {
                def cfApi = params.ENVIRONMENT == 'production' ? env.CF_API_PROD : env.CF_API_STAGING
                def cfOrg = params.ENVIRONMENT == 'production' ? env.CF_ORG_PROD : env.CF_ORG_STAGING
                def cfSpace = params.ENVIRONMENT == 'production' ? env.CF_SPACE_PROD : env.CF_SPACE_STAGING
                def appName = params.ENVIRONMENT == 'production' ? env.APP_NAME : "${env.APP_NAME}-staging"
                
                sh """
                    # Install CF CLI if not present
                    if ! command -v cf &> /dev/null; then
                        wget -q -O - https://packages.cloudfoundry.org/debian/cli.cloudfoundry.org.key | apt-key add -
                        echo "deb https://packages.cloudfoundry.org/debian stable main" | tee /etc/apt/sources.list.d/cloudfoundry-cli.list
                        apt-get update && apt-get install cf-cli
                    fi
                    
                    # Authenticate with Cloud Foundry
                    cf api ${cfApi}
                    cf auth ${CF_CREDS_USR} ${CF_CREDS_PSW}
                    cf target -o ${cfOrg} -s ${cfSpace}
                    
                    # Set environment variables for Artifactory
                    cf set-env ${appName} PIP_INDEX_URL "https://${ARTIFACTORY_CREDS_USR}:${ARTIFACTORY_CREDS_PSW}@${ARTIFACTORY_HOST}/api/pypi/pypi-virtual/simple/"
                    cf set-env ${appName} PIP_TRUSTED_HOST "${ARTIFACTORY_HOST}"
                    
                    # Deploy the application using the zip package
                    echo "Deploying ${appName} using package: dist/${APP_NAME}-${PACKAGE_VERSION}.zip"
                    cf push ${appName} -p dist/${APP_NAME}-${PACKAGE_VERSION}.zip
                    
                    # Verify deployment
                    cf app ${appName}
                    
                    echo "✅ Deployment to ${params.ENVIRONMENT} completed successfully"
                """
            }
        }
    }
}

post {
    always {
        echo '🧹 Cleaning up...'
        sh 'rm -rf venv/ || true'
        sh 'rm -rf .cache/ || true'
    }
    success {
        echo '✅ Pipeline completed successfully!'
        // Send success notification
        emailext (
            subject: "✅ Deployment Success: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
            body: "The deployment to ${params.ENVIRONMENT} was successful.\\n\\nBuild: ${env.BUILD_URL}",
            to: "${env.CHANGE_AUTHOR_EMAIL}"
        )
    }
    failure {
        echo '❌ Pipeline failed!'
        // Send failure notification
        emailext (
            subject: "❌ Deployment Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
            body: "The deployment to ${params.ENVIRONMENT} failed.\\n\\nBuild: ${env.BUILD_URL}\\nConsole: ${env.BUILD_URL}console",
            to: "${env.CHANGE_AUTHOR_EMAIL}"
        )
    }
}
```

}

# ===== CI/CD Environment Variables =====

# Set these in your GitLab CI/CD variables or Jenkins credentials:

# Artifactory Configuration

ARTIFACTORY_USER=your-username
ARTIFACTORY_TOKEN=your-api-token  # Mark as protected/masked
ARTIFACTORY_HOST=your-artifactory.company.com

# Cloud Foundry Configuration

CF_API_STAGING=https://api.cf-staging.company.com
CF_API_PROD=https://api.cf-prod.company.com
CF_ORG_STAGING=staging-org
CF_ORG_PROD=production-org
CF_SPACE_STAGING=development
CF_SPACE_PROD=production
CF_USERNAME=cf-deploy-user
CF_PASSWORD=cf-deploy-password  # Mark as protected/masked
CF_APPS_DOMAIN=apps.company.com

# ===== Local Testing Script =====

# test_package.sh - Test the package creation locally

#!/bin/bash
set -e

echo “🧪 Testing package creation locally…”

# Clean previous builds

rm -rf dist/ build/ *.egg-info/

# Create package (same command as CI/CD)

python setup.py sdist –formats=zip

# Verify package

echo “📦 Package created:”
ls -la dist/

echo “📋 Package contents:”
unzip -l dist/my-cf-app-1.0.0.zip

# Test installation in clean environment

echo “🧪 Testing package installation…”
python -m venv test_env
source test_env/bin/activate
pip install dist/my-cf-app-1.0.0.zip
python -c “import my_app; print(‘✅ Package installs correctly’)”
deactivate
rm -rf test_env

echo “✅ Package testing completed successfully!”
