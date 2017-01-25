total = 0
fibNo = 0 
fibPrev1 =1
fibPrev2 =0
while fibNo < 4000000:
	fibNo = fibPrev1 + fibPrev2
	fibPrev2 = fibPrev1
	fibPrev1 = fibNo
	if fibNo%2 == 0:
	   total = total + fibNo
print(total)

