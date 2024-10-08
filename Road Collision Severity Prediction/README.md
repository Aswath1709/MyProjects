CS6240_Project
Summer 2024

Author
-----------
- Joe Sackett (2018)
- Updated by Nikos Tziavelis (2023)
- Updated by Aswath Sundar (2024)
Installation
------------
These components are installed:
- OpenJDK 11
- Hadoop 3.3.5
- Maven (Tested with version 3.6.3)
- AWS CLI (Tested with version 1.22.34)

- Scala 2.12.17 (you can install this specific version with the Coursier CLI tool which also needs to be installed)
- Spark 3.3.2 (without bundled Hadoop)

After downloading the hadoop and spark installations, move them to an appropriate directory:

`mv hadoop-3.3.5 /usr/local/hadoop-3.3.5`

`mv spark-3.3.2-bin-without-hadoop /usr/local/spark-3.3.2-bin-without-hadoop`

Environment
-----------
1) Example ~/.bash_aliases:
	```
	export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
	export HADOOP_HOME=/usr/local/hadoop-3.3.5
	export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
	export SCALA_HOME=/usr/share/scala
	export SPARK_HOME=/usr/local/spark-3.3.2-bin-without-hadoop
	export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SCALA_HOME/bin:$SPARK_HOME/bin
	export SPARK_DIST_CLASSPATH=$(hadoop classpath)
	```

2) Explicitly set `JAVA_HOME` in `$HADOOP_HOME/etc/hadoop/hadoop-env.sh`:

	`export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64`

Execution
---------
1) Create a maven project called CS6240_Project
2) Download the scala files and put them in src/main/scala/com/example/
3) Update the pom.xml from here
4) Download the three csv files from the drive and upload it to an s3 bucket
5) Build the jar
6) Run the jar
7) To find the best parameters like performing grid search use randomForest.scala and decisionTree.scala
8) RF.scala is hardcoded with the best parameters for the model. Run RF.scala on AWS directly to fetch the prediction and metric values. 
