-- MySQL dump 10.13  Distrib 8.0.22, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: c9
-- ------------------------------------------------------
-- Server version	8.0.22

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `attend`
--

DROP TABLE IF EXISTS `attend`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `attend` (
  `id` int NOT NULL AUTO_INCREMENT,
  `SId` varchar(10) NOT NULL,
  `FId` varchar(10) NOT NULL,
  `CId` varchar(10) NOT NULL,
  `Dt` varchar(15) NOT NULL,
  `Class_attended` int NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=29 DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `attend`
--

LOCK TABLES `attend` WRITE;
/*!40000 ALTER TABLE `attend` DISABLE KEYS */;
INSERT INTO `attend` VALUES (1,'CSE18213','F01','15CSE311','2021-06-08',1),(2,'CSE18213','F01','15CSE311','2021-06-09',1),(3,'CSE18213','F01','15CSE311','2021-06-11',0),(4,'CSE18213','F01','15CSE311','2021-06-12',1),(5,'CSE18213','F01','15CSE312','2021-06-08',1),(6,'CSE18213','F01','15CSE312','2021-06-09',1),(7,'CSE18213','F01','15CSE312','2021-06-10',0),(8,'CSE18214','F01','15CSE311','2021-06-08',1),(9,'CSE18214','F01','15CSE311','2021-06-09',0),(10,'CSE18214','F01','15CSE311','2021-06-11',0),(11,'CSE18214','F01','15CSE311','2021-06-12',0),(12,'CSE18214','F01','15CSE312','2021-06-08',1),(13,'CSE18214','F01','15CSE312','2021-06-09',1),(14,'CSE18214','F01','15CSE312','2021-06-10',1),(15,'CSE18213','F01','15CSE311','2021-06-13',1),(16,'CSE18214','F01','15CSE311','2021-06-13',0),(17,'CSE18213','F01','15CSE311','2021-06-13',1),(18,'CSE18214','F01','15CSE311','2021-06-13',0),(19,'CSE18213','F01','15CSE311','2021-06-13',1),(20,'CSE18214','F01','15CSE311','2021-06-13',0),(21,'CSE18214','F01','15CSE311','2021-06-13',0),(22,'CSE18213','F01','15CSE311','2021-06-13',1),(23,'CSE18213','F01','15CSE311','2021-06-13',0),(24,'CSE18214','F01','15CSE311','2021-06-13',1),(25,'CSE18213','F01','15CSE311','2021-06-13',0),(26,'CSE18214','F01','15CSE311','2021-06-13',0),(27,'CSE18214','F01','15CSE311','2021-06-13',0),(28,'CSE18213','F01','15CSE311','2021-06-13',0);
/*!40000 ALTER TABLE `attend` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `attendance`
--

DROP TABLE IF EXISTS `attendance`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `attendance` (
  `SId` varchar(10) NOT NULL,
  `FId` varchar(10) NOT NULL,
  `CId` varchar(10) NOT NULL,
  `Class_attended` int NOT NULL,
  `Total_classes` int NOT NULL,
  PRIMARY KEY (`SId`,`FId`,`CId`),
  KEY `fk_AttendanceCourse` (`CId`),
  KEY `fk_AttendanceFaculty` (`FId`),
  CONSTRAINT `fk_AttendanceCourse` FOREIGN KEY (`CId`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_AttendanceFaculty` FOREIGN KEY (`FId`) REFERENCES `faculty` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_AttendanceStudent` FOREIGN KEY (`SId`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `attendance`
--

LOCK TABLES `attendance` WRITE;
/*!40000 ALTER TABLE `attendance` DISABLE KEYS */;
INSERT INTO `attendance` VALUES ('CSE18213','F01','15CSE311',8,12),('CSE18213','F01','15CSE312',2,3),('CSE18214','F01','15CSE311',2,12),('CSE18214','F01','15CSE312',3,3);
/*!40000 ALTER TABLE `attendance` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `course`
--

DROP TABLE IF EXISTS `course`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `course` (
  `id` varchar(10) NOT NULL,
  `C_Name` varchar(20) NOT NULL,
  `C_Sem` int NOT NULL,
  `C_type` varchar(8) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `course`
--

LOCK TABLES `course` WRITE;
/*!40000 ALTER TABLE `course` DISABLE KEYS */;
INSERT INTO `course` VALUES ('15CSE311','Compiler Design',6,'Core'),('15CSE312','Networks',6,'Core'),('15HUM240','Psychology',6,'Elective');
/*!40000 ALTER TABLE `course` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `f_email`
--

DROP TABLE IF EXISTS `f_email`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `f_email` (
  `id` varchar(20) NOT NULL,
  `email` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `foreignkey` FOREIGN KEY (`id`) REFERENCES `faculty` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `f_email`
--

LOCK TABLES `f_email` WRITE;
/*!40000 ALTER TABLE `f_email` DISABLE KEYS */;
INSERT INTO `f_email` VALUES ('F01','abishekraguram@gmail.com'),('F02','abishekraguram@gmail.com'),('F04','manomeha2001@gmail.com');
/*!40000 ALTER TABLE `f_email` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `faculty`
--

DROP TABLE IF EXISTS `faculty`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `faculty` (
  `id` varchar(10) NOT NULL,
  `F_Name` varchar(20) NOT NULL,
  `F_Username` varchar(20) NOT NULL,
  `F_Password` varchar(20) NOT NULL,
  `F_Dept` varchar(20) NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `faculty`
--

LOCK TABLES `faculty` WRITE;
/*!40000 ALTER TABLE `faculty` DISABLE KEYS */;
INSERT INTO `faculty` VALUES ('F01','Padmavathi','padhu','sabarishiloveu','CSE'),('F02','Rajalakshmi','raja','raja','HUM'),('F03','Sivashankar','sivas','sivas','CSE'),('F04','Sivashankar','sivas','sivas','CSE');
/*!40000 ALTER TABLE `faculty` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `msg`
--

DROP TABLE IF EXISTS `msg`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `msg` (
  `MsgId` int NOT NULL AUTO_INCREMENT,
  `SId` varchar(20) NOT NULL,
  `FId` varchar(20) NOT NULL,
  `Msg` varchar(200) NOT NULL,
  `Ack` int NOT NULL,
  `Doc` longtext,
  `Dt` varchar(20) NOT NULL,
  PRIMARY KEY (`MsgId`),
  UNIQUE KEY `MsgId_UNIQUE` (`MsgId`),
  KEY `MessageStudent_idx` (`SId`),
  KEY `MessageFaculty_idx` (`FId`),
  CONSTRAINT `MessageFaculty` FOREIGN KEY (`FId`) REFERENCES `faculty` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `MessageStudent` FOREIGN KEY (`SId`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=24 DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `msg`
--

LOCK TABLES `msg` WRITE;
/*!40000 ALTER TABLE `msg` DISABLE KEYS */;
INSERT INTO `msg` VALUES (1,'CSE18213','F01','Hi aunty epdi irukinga',1,NULL,'01/06/2021'),(2,'CSE18213','F01','Hi padhu epdi iruka',1,NULL,'01/06/2021'),(3,'CSE18213','F01','Sappa epdi irukaan',0,NULL,'02/05/2020'),(4,'CSE18213','F01','askjfnljanslf',0,NULL,'01/06/2021'),(17,'CSE18213','F01','asjfnlnldkg',0,NULL,'2021-06-3'),(18,'CSE18213','F01','Mail test',0,NULL,'2021-06-11'),(19,'CSE18213','F01','Mail test 2',0,NULL,'2021-06-11'),(20,'CSE18213','F01','mail test 3',0,NULL,'2021-06-11'),(21,'CSE18213','F01','test mail',0,NULL,'2021-06-11'),(22,'CSE18213','F01','test email',0,NULL,'2021-06-11'),(23,'CSE18213','F01','test email 7',0,NULL,'2021-06-11');
/*!40000 ALTER TABLE `msg` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `s_email`
--

DROP TABLE IF EXISTS `s_email`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `s_email` (
  `id` varchar(20) NOT NULL,
  `email` varchar(150) NOT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `foreignkey1` FOREIGN KEY (`id`) REFERENCES `student` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `s_email`
--

LOCK TABLES `s_email` WRITE;
/*!40000 ALTER TABLE `s_email` DISABLE KEYS */;
INSERT INTO `s_email` VALUES ('CSE18203','abishekvasanth@gmail.com'),('CSE18213','manomeha@gmail.com'),('CSE18214','abishekraguram@gmail.com');
/*!40000 ALTER TABLE `s_email` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Table structure for table `student`
--

DROP TABLE IF EXISTS `student`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `student` (
  `id` varchar(10) NOT NULL,
  `S_Name` varchar(20) NOT NULL,
  `S_Username` varchar(20) NOT NULL,
  `S_Password` varchar(20) NOT NULL,
  `C_elective` varchar(10) DEFAULT NULL,
  `S_Dept` varchar(20) NOT NULL,
  `S_Sec` varchar(5) NOT NULL,
  `S_Sem` int DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_CourseStudent` (`C_elective`),
  CONSTRAINT `fk_CourseStudent` FOREIGN KEY (`C_elective`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `student`
--

LOCK TABLES `student` WRITE;
/*!40000 ALTER TABLE `student` DISABLE KEYS */;
INSERT INTO `student` VALUES ('CSE18203','Absihek','abis','abis',NULL,'CSE','C',6),('CSE18213','Aswanth','ashrag','gayathri','15HUM240','CSE','C',6),('CSE18214','Aswath Sundar','aswath','aswath',NULL,'CSE','C',6);
/*!40000 ALTER TABLE `student` ENABLE KEYS */;
UNLOCK TABLES;

--
-- Dumping events for database 'c9'
--

--
-- Dumping routines for database 'c9'
--
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2021-06-11 18:25:57
