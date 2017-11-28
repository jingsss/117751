
use strict;
chdir('openSMILE-2.1.0');
# --------------------------------------------------------------------
# translate label to a number
my %emo_trans = '';
$emo_trans{'N'} = 1;
$emo_trans{'J'} = 2;
$emo_trans{'S'} = 3;
$emo_trans{'F'} = 4;
$emo_trans{'A'} = 5;
$emo_trans{'C'} = 6;
$emo_trans{'D'} = 7;

my $loc = '../combine_data/train';
my @lines = <$loc/*>;
my $count = 0;
my $header = 'openSMILE-2.3.0';
if (-e 'Features/train.arff') {system("rm Features/train.arff")}
foreach my $line (@lines) {
	if ($line =~ 'N' || $line =~ 'J' || $line =~ 'S'|| $line =~ 'F'|| $line =~ 'A'|| $line =~ 'C'|| $line =~ 'D'){
#		 print "$line\n";
		$_ = $line;
		s/(N|J|S|F|A|C|D)//;
		my $emo = $emo_trans{$&};
		
		my $cmd = "SMILExtract -C config/emo_IS09.conf -I $line -O ../Features/train.arff -classes {1,2,3,4,5,6,7} -classlabel $emo";
		$count++;
		print "$cmd\n";
		system($cmd);
	}
}

my @names = ('train'); 
foreach my $name (@names) {
	open(INFO, "../Features/$name.arff");
	my @lines = <INFO>;
	close(INFO);
	@lines[387] = "\@attribute class \{1,2,3,4,5,6,7\}\n";

	open(SOURCE, ">../Features/$name.arff");
	print SOURCE @lines;
	close(SOURCE);
}