-- --------------------------------------------------------

-- 
-- Table structure for table `alternatives`
-- 

CREATE TABLE `alternatives` (
  `ID` int(10) unsigned NOT NULL auto_increment,
  `QID` int(10) unsigned NOT NULL default '0',
  `Label` text NOT NULL,
  `Ordering` int(10) unsigned NOT NULL default '0',
  PRIMARY KEY  (`ID`)
) TYPE=MyISAM AUTO_INCREMENT=1 ;

