#Our results show that using our features together with MFCCs and pitch related features lead to a better performance.
#http://emotion-research.net/sigs/speech-sig/emotion-challenge/INTERSPEECH-Emotion-Challenge-2009_draft.pdf
# compare with 384, http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7289391
import weka
import weka.core.jvm as jvm
jvm.start()
data_dir = "/my/datasets/"