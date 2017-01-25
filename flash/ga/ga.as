class game
{
	static function restart() 
	{
		_root.player.kill();
		_root.player=undefined;
	}
	
	static function restartga() 
	{
		trace('------------------');
		trace('RESTARTED GA');
		_root.player.kill();
		_root.player=undefined;
		_root.ga=new gaHandler();
	}
	
	// Startup Function
	static function main() 
	{
		//Stage.scaleMode = "noScale";
		
		console.startup();
				
		//background
		plot.rect(_root,0,0,499,399,0xFFFFFF);
		
		//some numbers
		_root.size=new cartesian(500, 400);
		_root.center=new cartesian(250, 200);
		_root.grav=new cartesian(0, 0.3);

		//Start the game!
		_root.key=new keyHandler();
		_root.mouse=new mouseHandler();
		_root.cam=new cameraHandler();
		_root.sectors=new sectorHandler();
		_root.player=undefined;
		_root.ga=new gaHandler();
		_root.gui=new guiHandler();
		_root.timer=0;
		_root.gamePause=false;
		_root.dialog=false;
		_root.tools=new toolHandler();
		
		_root.sectors=parser.loadMap('-18 1i 18 1i 84 1i ai n bl 11 dh k ea 1b fi 13 gq n i4 11 j2 p k0 8 l6 a lq p mq 16 nj 11 om f pk 3 rb a s4 a sg a t2 -5 tj -h ur -a v8 8 101 p 114 k 122 u 12t 13 13r p 14p 5 15q -7 16e 0 172 -7 17t -c 18k -2 1al -2 1bq -k 1ct -15 1dr -10 1e5 -r 1en -1a 1gb -1n 1h4 -p 1hm 0 1if 5 1j5 3 1k6 -c 1kv 5 1lj i 1n5 21,-18 1i -au 1i -gs 1i,1n5 21 1nf 21 1ok 1n 1pd 1i 1qg 11 1r4 1d 1sc 13 1t5 u 1v9 0,212 -f 21k -h 22k -10 23l -1f 24q -10 25c -r 25r -r 26f -15 27a -1i 286 -1a 28q -k 2a4 k 2b5 s 2ca k 2dn -a 2fe -r 2g0 -f 2h3 -10 2i1 -1f 2jg -p 2k9 -7 2lp d 2o1 1g 2og 2g 2p2 2q 2r3 26 2ru 1q 2tt 1s 2v0 1s 303 1b 313 n 32q p 34u u 35u 0 36q -m 37t -u 38h -u 39c -1a 39u -1k 3b1 -2n 3cg -3l 3dh -2n 3gv -1a 3jh -1a 3ll -21 3o0 -44 3ou -50 3qi -4t 3rq -42 3t2 -3o 3up -49,3up -49 3vq -31 40q -2d 42k -26 45l -1a 47a k 488 1i 4ao f 4c5 -a 4e6 3 4fb -7 4h2 -10 4ii -1n 4kj -1d 4lh -1d 4mu -2i 4ns -31 4pt -49 4rr -39 4t3 -34 4tn -2s 4vr -1u 505 -1a 515 -1s 51i -1p 526 -1i 52l -1p 53e -1s 53q -1a 54h -1f 557 -1p 565 -15 56f -p 573 -f 592 1s 5ac 2o 5cb 2v 5ee 2e 5g5 1i 5gn 2g 5h8 1d 5i1 1g 5ie 1n 5jh 16 5kp p 5m1 -c,5nt -1f 5oo -10 5pp -1d 5qn -1k 5sl n 5uu 13 60l n 619 1g 611 21 62c 21 630 2g 62j 39 63m 37 64i 39 66o 37 68f 2v 6ad 2j 6be 2v 6ce 3m 6dk 3h 6gj 1s 6ic i 6kl -2 6lb p 6lv d 6me 0 6nj d 6o7 f 6o3 4f 6p3 50 6ql 4a 6rh 4a 6sc 4r 6ta 53 6vt 68 72a 6d 74b 63 76m 5n 783 55 79q 53 7bh 4f 7d1 2j 7ee f 7hc -p 7iu a 7ke s 7nq 2q 7pj 1v 7rs 1i 7um 1n,5lv -b 5nt -1f,1v7 1 212 -e##');
		
		//Main loop
		_root.onEnterFrame=function()
		{
			_root.key.update();
			_root.mouse.update();
			
			
			if(!_root.gamePause)
			{
				_root.ga.update();
				_root.player.update();
				_root.ga.update();
				_root.player.update();
			}
			
			_root.tools.update();
			_root.gui.update();
			
			_root.cam.update();
			_root.sectors.draw();
			_root.player.draw();
			_root.tools.draw();
		}
	}
}