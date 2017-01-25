-- --------------------------------------------------------

-- 
-- Table structure for table `users`
-- 

CREATE TABLE `users` (
  `ID` int(10) unsigned NOT NULL auto_increment,
  `Sex` varchar(45) NOT NULL default '',
  `Start` datetime NOT NULL default '0000-00-00 00:00:00',
  `End` datetime NOT NULL default '0000-00-00 00:00:00',
  PRIMARY KEY  (`ID`)
) ENGINE=MyISAM AUTO_INCREMENT=1 ;