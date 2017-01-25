import unittest, grn, random

class TestSequenceFunctions(unittest.TestCase):
    
    def setUp(self):
        random.seed(2)
        self.DELTA = 1
        self.names = ["tests/eoinseed1.txt","tests/eoinseed2.txt",
                 "tests/eoinseed3.txt","tests/eoinseed4.txt",
                 "tests/eoinseed5.txt","tests/eoinseed6.txt",
                 "tests/eoinseed7.txt","tests/eoinseed8.txt",
                 "tests/eoinseed9.txt","tests/eoinseed10.txt"]
        self.grn = grn.GRN()
        
    def test_parsing(self):
        self.failIf(self.grn.sensible_parsing)

    def test_eoin(self):
        sizes = []
        final_concs = []
        correct_sizes = [7, 8, 7, 7, 8, 9, 10, 10, 7, 8]
        correct_concs = [0.10536119562462604, 1.0785531576949638e-05, 0.96204392235109859, 0.036227250980492995, 0.14936626819446591, 0.0017180411368314544, 0.74527253618090816, 0.20216842837768173, 1.8597966179449688e-41, 9.8837209610607373e-11, 2.8433950003789704e-12, 3.7272818452249546e-19, 0.79783157161841423, 0.99999999990116273, 1.0605769176570928e-12, 0.43021628377545634, 1.8607359883012191e-14, 7.8687897480028881e-11, 0.10528617335121833, 0.99999999999998135, 1.1568225120729014e-75, 0.46449754279463745, 1.0, 1.0, 3.852020054491351e-50, 2.2676764109720152e-47, 1.2698664131258545e-50, 4.614598027941728e-51, 1.0230365114501557e-49, 0.96250388634805484, 0.10434327320808674, 9.8246257052620553e-11, 9.8246257052620553e-11, 0.89565672649717465, 0.015449167345545771, 9.8246257052620553e-11, 0.022046946306399435, 1.549088544201994e-51, 2.0653767133682643e-49, 3.1437911279343385e-51, 0.18145538980022971, 1.0, 5.5872364375449987e-50, 0.1892264857816604, 0.62931812441810986, 5.4289702573481519e-52, 0.00076655281898026532, 0.99999999960006059, 0.0021459130763556624, 9.9984846630215615e-11, 0.75450804442096187, 0.21977813174014549, 9.9984846630215615e-11, 9.9984846630215615e-11, 9.9984846630215615e-11, 0.022801357943556831, 0.016570752661809442, 0.006791886227358491, 0.012708024669514883, 0.0055977427316390221, 0.97053528744991513, 0.0022669597556181486, 9.3192992915059737e-11, 0.0061020387097514761, 3.4951165492265383e-08, 0.97942727275003505, 0.17202757039233571, 0.99999999989999022, 1.0000982013359345e-10, 0.00012387320936094268, 0.77259278064842241, 0.055248455240959803, 7.3205089211345301e-06, 4.4798299830043145e-05, 0.10714347255184113, 0.83990369807369525, 9.9719383174870601e-11, 0.16009630172686587, 5.2259906141012131e-126, 9.9719383174870601e-11, 0.89281172914832874]

        for name in self.names:
            self.grn = grn.GRN(delta=self.DELTA)
            self.grn.read_genome(name)
            self.grn.build_genes()
            self.grn.precalc_matrix()
            self.grn.regulate_matrix(2000)
            sizes.append(len(self.grn.conc_list))
        
            for conc in self.grn.conc_list:
                final_concs.append(conc[-1])

        for i in range(len(sizes)):
            self.assertEqual(sizes[i], correct_sizes[i])
        for i in range(len(final_concs)):
            self.assertEqual(final_concs[i], correct_concs[i])

    def test_miguel(self):
        concs = []
        correct_concs = [2.4694881581878343e-12, 1.1537420071432216e-12, 2.3929814982520794e-37, 0.79999999999637683, 0.05, 0.05, 0.05, 0.05]

        self.grn = grn.GRN(1)
        self.grn.read_genome("tests/miguel_parsed.txt")
        self.grn.build_genes()  
        self.grn.add_extra("EXTRA_A", 0.05, [0]*32)
        self.grn.add_extra("EXTRA_B", 0.05, [1]*32)
        self.grn.add_extra("EXTRA_C", 0.05, [0]*16 +[1]*16)
        self.grn.add_extra("EXTRA_D", 0.05, [1]*16 +[0]*16)
        self.grn.precalc_matrix()
        size = len(self.grn.conc_list)
        self.assertEqual(size, 8)
        self.grn.regulate_matrix(1000)
        
        for conc in self.grn.conc_list:
            concs.append(conc[-1])
        for i in range(len(concs)):
            self.assertEquals(concs[i], correct_concs[i])

        
            
    def tearDown(self):
        self.grn = None

def main():
    unittest.main()

if __name__ == '__main__':
    main()
