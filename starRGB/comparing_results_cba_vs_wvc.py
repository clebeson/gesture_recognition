import numpy as np

cba = np.array([69, 80,78,79,95,86,91,84,90,75,67,82,94,79,70,94,91,81,89,87])
wvc = np.array([75, 87.91, 91.57, 91.1, 95.95, 85.86, 95.71, 89.19, 95.65, 89.41, 81.03, 89.36, 97.24, 88.04,73.84,96.22,93.06,88.76,97.22,92])
class_names = ['prendere', 'vieniqui', 'perfetto', 'fame', 'sonostufo', 'seipazzo', 'basta', 'cheduepalle', 'noncenepiu', 'chevuoi',
                'ok', 'combinato', 'freganiente', 'cosatifarei', 'buonissimo', 'vattene', 'messidaccordo', 'daccordo', 'furbo', 'tantotempo']
result = ["\\textit({})  & {:.2f}\\% ".format(cn,(float(w-c)/c)*100) for w,c,cn in zip(wvc,cba, class_names)]

print(" & ".join(result))
print(cba.min(), cba.max(), wvc.min(),wvc.max())
