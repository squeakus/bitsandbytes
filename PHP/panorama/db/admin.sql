-- --------------------------------------------------------

-- 
-- Table structure for table `admin`
-- 

CREATE TABLE `admin` (
  `ID` int(10) unsigned NOT NULL auto_increment,
  `login` varchar(45) NOT NULL default '',
  `password` varchar(45) NOT NULL default '',
  PRIMARY KEY  (`ID`)
) ENGINE=MyISAM AUTO_INCREMENT=1 ;

INSERT INTO `admin` VALUES (1, 'backweb', 'backweb');
INSERT INTO `admin` VALUES (2, 'chalmers', 'panorama');
