
DROP DATABASE IF EXISTS uvprojet1;

CREATE DATABASE uvprojet1;
use uvprojet1;




CREATE TABLE `etudiant` (
  `id` int(11) NOT NULL UNIQUE AUTO_INCREMENT,
  `nom` varchar(100) NOT NULL,
  `prenom`  varchar(100) NOT NULL,
  `option` varchar(100) NOT NULL,
  `niveau` varchar(100) NOT NULL,
  `matricule` varchar(100) NOT NULL UNIQUE,
  `present` varchar(100) NOT NULL DEFAULT 'NO',
 
   PRIMARY KEY (`id`)

) ENGINE=InnoDB DEFAULT CHARSET=latin1;



INSERT INTO `etudiant` (`id`,`nom`,`prenom`,`option`,`niveau`,`matricule`,`present`) VALUES
(1,'Fotso','Yvanol','IA','IV','cm-uds-19sci1710','NO'),
(2,'Fotso 2 ','Yvanol 2 ','IA','IV','cm-uds-20sci1004','NO'),
(3,'Tankou','Ghislain','IA','IV','cm-uds-20sci1002','NO'),
(4,'Fokou','Laures','IA','IV','cm-uds-22sci1155','NO'),
(5,'Ousmane','Ousmane','RSD','IV','cm-uds-20sci1003','NO'),
(6,'Ariane','Ariane','IA','IV','cm-uds-20sci1001','NO'),
(7,'Abdel','Abdel','IA','IV','cm-uds-20sci1000','NO'),
(8,'Romuald','Romuald','IA','IV','cm-uds-20sci1005','NO');








CREATE TABLE `admin` ( 
  `id` int(11) NOT NULL UNIQUE AUTO_INCREMENT,
  `email` varchar(100) NOT NULL UNIQUE,
  `password` varchar(100) NOT NULL,

   PRIMARY KEY (`id`)

) ENGINE=InnoDB DEFAULT CHARSET=latin1;



INSERT INTO `admin` (`id`,`email`,`password`) VALUES
(1,'fotso@gmail.com','uvprojet12345'),
(2,'tankou@gmail.com','uvprojet12345');








CREATE TABLE `chef_salle` ( 
  `id` int(11) NOT NULL UNIQUE AUTO_INCREMENT,
  `email` varchar(100) NOT NULL UNIQUE,
  `password` varchar(100) NOT NULL,

   PRIMARY KEY (`id`)

) ENGINE=InnoDB DEFAULT CHARSET=latin1;


INSERT INTO `chef_salle` (`id`,`email`,`password`) VALUES
(1,'fotso@gmail.com','uvprojet12345'),
(2,'tankou@gmail.com','uvprojet12345');








-- une matiere peut passer dans une ou plusieur salle et dans une salle on peut faire passer une ou plusieur matiere //POUR USE UN CHAMP COMME Foreign Key il doit etre UNIQUE dans sa table d'origine


CREATE TABLE `matiere` ( 
  `id` int(11) NOT NULL UNIQUE AUTO_INCREMENT,
  `matricule_etudiant_mat` varchar(100) NOT NULL,
  `nom` varchar(100) NOT NULL,
  `code` varchar(100) NOT NULL,
  `heure` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,

   PRIMARY KEY (`id`),
   CONSTRAINT fkamat FOREIGN KEY (`matricule_etudiant_mat`) REFERENCES etudiant(`matricule`)

) ENGINE=InnoDB DEFAULT CHARSET=latin1;


INSERT INTO `matiere` (`id`,`matricule_etudiant_mat`,`nom`,`code`,`heure`) VALUES
(1,'cm-uds-19sci1710','Genie Logiciel II','INF 427','2023-06-13 10:00:00'),
(2,'cm-uds-20sci1004','Intergiciel','RSD 418','2023-06-15 12:00:00'),
(3,'cm-uds-20sci1002','Compilation','INF 428','2023-06-17 14:00:00'),
(4,'cm-uds-22sci1155','Intergiciel','RSD 418','2023-06-19 16:00:00'),
(5,'cm-uds-20sci1003','Fouille de donnee','RSD 418','2023-06-21 18:00:00'),
(6,'cm-uds-20sci1001','Fouille de donnee','RSD 418','2023-06-23 18:00:00'),
(7,'cm-uds-20sci1000','Fouille de donnee','RSD 418','2023-06-25 18:00:00'),
(8,'cm-uds-20sci1005','Fouille de donnee','RSD 418','2023-06-27 18:00:00');









CREATE TABLE `salle` ( 
  `id` int(11) NOT NULL UNIQUE AUTO_INCREMENT,
  `matricule_etudiant_sal` varchar(100)  NOT NULL,
  `nom` varchar(100) NOT NULL,
  `code` varchar(100) NOT NULL,
 
   PRIMARY KEY (`id`),
   CONSTRAINT fkasal FOREIGN KEY (`matricule_etudiant_sal`) REFERENCES etudiant(`matricule`)

) ENGINE=InnoDB DEFAULT CHARSET=latin1;


INSERT INTO `salle` (`id`,`matricule_etudiant_sal`,`nom`,`code`) VALUES
(1,'cm-uds-19sci1710','Amphi','Amphi 600'),
(2,'cm-uds-20sci1004','Salle Jumelle','Annexe 600'),
(3,'cm-uds-20sci1002','Salle NS','NS 2'),
(4,'cm-uds-22sci1155','Salle NS','NS 3'),
(5,'cm-uds-20sci1003','Salle NS','NS 1'),
(6,'cm-uds-20sci1001','Salle','Japonaise 1'),
(7,'cm-uds-20sci1000','Amphi','Amphi 1000 '),
(8,'cm-uds-20sci1005','Salle','Japonaise 3');




