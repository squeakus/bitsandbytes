-- --------------------------------------------------------

-- 
-- Table structure for table `questions`
-- 

CREATE TABLE `questions` (
  `ID` int(10) unsigned NOT NULL auto_increment,
  `SelectType` tinyint(3) unsigned NOT NULL default '0',
  `Label` text NOT NULL,
  `Ordering` int(10) unsigned NOT NULL default '0',
  PRIMARY KEY  (`ID`)
) TYPE=MyISAM AUTO_INCREMENT=1 ;

