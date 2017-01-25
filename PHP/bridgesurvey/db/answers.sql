-- 
-- Table structure for table `answers`
-- 

CREATE TABLE `answers` (
  `ID` int(10) unsigned NOT NULL auto_increment,
  `FRONT` tinyint(4) NOT NULL,
  `BACK` tinyint(4) NOT NULL,
  `NOPREF` tinyint(4) NOT NULL,
  PRIMARY KEY  (`ID`)
) TYPE=MyISAM AUTO_INCREMENT=1 ;

