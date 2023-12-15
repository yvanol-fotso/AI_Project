
DROP DATABASE IF EXISTS uvprojet1;

CREATE DATABASE uvprojet1;
use uvprojet1;




CREATE TABLE `student1` (
  `id` int(11) NOT NULL UNIQUE AUTO_INCREMENT,
  `departement` varchar(40) NOT NULL,
  `cour` varchar(40) NOT NULL,
  `annee` varchar(40) NOT NULL,
  `semestre` varchar(40) NOT NULL,
  `id_etudiant` varchar(40) NOT NULL,
  `nom_etudiant` varchar(40) NOT NULL,
  `division` varchar(40) NOT NULL,
  `rol_numero` varchar(40) NOT NULL,
  `genre` varchar(40) NOT NULL,
  `date`  timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `email` varchar(40) NOT NULL,
  `phone` varchar(40) NOT NULL,
  `address` varchar(40) NOT NULL,
  `teacher` varchar(40) NOT NULL,
  `bouton_radio` varchar(40) NOT NULL,
  
   PRIMARY KEY (`id`)

) ENGINE=InnoDB DEFAULT CHARSET=latin1;





CREATE TABLE `admin` (
  `id` int(11) NOT NULL UNIQUE AUTO_INCREMENT,
  `email` varchar(100) NOT NULL UNIQUE,
  `password` varchar(100) NOT NULL,

   PRIMARY KEY (`id`)

) ENGINE=InnoDB DEFAULT CHARSET=latin1;


INSERT INTO `admin` (`id`,`email`,`password`) VALUES(1,"fotso@gmail.com","uvprojet12345");
