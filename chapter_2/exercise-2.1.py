# given in ques
p_cancer = 0.001
p_no_cancer = 1 - p_cancer
p_pos_given_cancer = 0.9 # true positive rate
p_pos_given_no_cancer = 0.03 # false positive rate

# Bayes' theorem
p_cancer_given_pos = (p_pos_given_cancer * p_cancer) / (p_pos_given_cancer * p_cancer + p_pos_given_no_cancer * p_no_cancer)

print(f"Probability of having cancer given a positive test result, p(C=1|T=1): {p_cancer_given_pos:.4f} or {p_cancer_given_pos*100:.2f}%")

