#!/bin/bash

# Function to get Java executable path
get_java_exec() {
  if [[ -n "$JAVA_HOME" ]]; then
    echo "$JAVA_HOME/bin/java"
  else
    echo "java"
  fi
}

# Function to extract the major version of Java
get_java_major_version() {
  local version=$($1 -version 2>&1 | awk -F '"' '/version/ {print $2}')
  echo ${version%%.*}
}

# Verify Java version is JDK 25 or newer
JAVA=$(get_java_exec)
JAVA_MAJOR_VERSION=$(get_java_major_version $JAVA)
if [[ "$JAVA_MAJOR_VERSION" -lt 25 ]]; then
  echo "Error: JDK 25 or newer is required to run this application."
  exit 1
fi

# Define the path of the relative JAR
BI4J_RELATIVE_JAR="./bitubi-inference-4j-cli/target/bitubi-inference-4j-cli.jar"
# Path to the logback.xml
LOGBACK_CONFIG="./conf/logback.xml"

BI4J_JVM_ARGS="$BI4J_JVM_ARGS -ea -server -Dstdout.encoding=UTF-8 -Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0 --add-opens=jdk.incubator.vector/jdk.incubator.vector=ALL-UNNAMED --add-modules=jdk.incubator.vector --add-exports java.base/sun.nio.ch=ALL-UNNAMED --enable-preview --enable-native-access=ALL-UNNAMED \
 -XX:+UnlockDiagnosticVMOptions -XX:CompilerDirectivesFile=./inlinerules.json -XX:+AlignVector -XX:+UseStringDeduplication \
 -XX:+UseCompressedOops -XX:+UseCompressedClassPointers"
        
# Check if PREINSTALLED_JAR environment variable is set
if [[ -z "$BI4J_PREINSTALLED_JAR" ]]; then
  # If the relative JAR doesn't exist, build it
  if [[ ! -f $BI4J_RELATIVE_JAR ]]; then
    echo "The JAR $BI4J_RELATIVE_JAR is missing. Attempting to build..."
    ./mvnw clean package -DskipTests
    if [[ $? -ne 0 ]]; then
      echo "Error building the JAR. Please check your build setup."
      exit 1
    fi
  fi
  # Run the JAR in a relative directory
  $JAVA $BI4J_JVM_ARGS $BI4J_JVM_ARGS_EXTRA  -jar $BI4J_RELATIVE_JAR "$@"
else
  # If PREINSTALLED_JAR is set, run the JAR specified by the variable
  $JAVA $BI4J_JVM_ARGS $BI4J_JVM_ARGS_EXTRA -jar $BI4J_PREINSTALLED_JAR "$@"
fi
