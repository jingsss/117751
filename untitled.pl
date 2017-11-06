
use strict;
chdir('openSMILE-2.2rc1');
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

my $loc = '../audio_segment/ice_age_a_mammoth_christmas/';
my @lines = <$loc/*>;
my $count = 0;

foreach my $line (@lines) {
	if ($line =~ 'N' || $line =~ 'J' || $line =~ 'S'|| $line =~ 'F'|| $line =~ 'A'|| $line =~ 'C'|| $line =~ 'D'){
		# print "$line\n";
		$_ = $line;
		s/(N|J|S|F|A|C|D)//;
		my $emo = $emo_trans{$&};
		# print "$emo\n";
		my $cmd = "SMILExtract -C config/emo_IS09.conf -I $line -O ../Features/temp.arff -instname $count -classes {1,2,3,4,5,6,7} -classlabel $emo";
		$count++;
	}
}