CREATE TABLE `freetextanswers` (
  `ID` int(10) unsigned NOT NULL auto_increment,
  `UID` int(10) unsigned NOT NULL default '0',
  `QID` int(10) unsigned NOT NULL default '0',
  `answer` text NOT NULL,
  PRIMARY KEY  (`ID`),
  KEY `FK_freeTextAnswers_UID` (`UID`),
  KEY `FK_freeTextAnswers_QID` (`QID`)
) ENGINE=MyISAM AUTO_INCREMENT= 1;
