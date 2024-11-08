function [gmean] = Evaluate(ACTUAL,PREDICTED,pos_class)
idx = (ACTUAL()==pos_class);
p = length(ACTUAL(idx));
n = length(ACTUAL(~idx));
N = p+n;
tp = sum(ACTUAL(idx)==PREDICTED(idx));
tn = sum(ACTUAL(~idx)==PREDICTED(~idx));
fp = n-tn;
tp_rate = tp/p;
recall = tp_rate;
precision = tp/(tp+fp);
gmean = sqrt(precision*recall);

end
