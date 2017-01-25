-- --------------------------------------------------------

-- 
-- Table structure for table `answers`
-- 

CREATE TABLE `answers` (
  `ID` int(10) unsigned NOT NULL auto_increment,
  `UID` int(10) unsigned NOT NULL default '0',
  `QID` int(10) unsigned NOT NULL default '0',
  `AID` int(10) unsigned NOT NULL default '0',
  PRIMARY KEY  (`ID`)
) TYPE=MyISAM AUTO_INCREMENT=1 ;

