node {
   def cmake
   stage('Preparation') { // for display purposes
      // Get some code from a GitHub repository
      git credentialsId: 'creds', url: '.git'

      cmake = "/usr/bin/cmake"
   }
  stage('Build') {
    dir('sample/build') {
        deleteDir()
        writeFile file:'dummy', text:'' 
            sh 'cmake ../'
            sh 'make'
    }
  }
  stage('Results') {
      dir('sample/build') {
      sh 'for f in *test; do ./$f --gtest_output=xml:$f.xml; done'
      junit '*test.xml'
      }
  }
}
