ones = {
	1:'one',
	2:'two',
	3:'three',
	4:'four',
	5:'five',
	6:'six',
	7:'seven',
	8:'eight',
	9:'nine',
}
teens = {
	3:'thir',
	4:'four',
	5:'fif',
	6:'six',
	7:'seven',
	8:'eigh',
	9:'nine',
}
tens = {
	2:'twen',
	3:'thir',
	4:'for',
	5:'fif',
	6:'six',
	7:'seven',
	8:'eigh',
	9:'nine',
}
count = 0
for i in range(1, 1000):
	h = i // 100
	t = (i % 100)//10
	o = i % 10
	word = ''
	if h != 0:
		word += ones[h] + ' hundred'
		if t != 0 or o != 0:
			word += ' and '
	if t != 0:
		if t == 1:
			if o == 0:
				word += 'ten'
			elif o == 1:
				word += 'eleven'
			elif o == 2:
				word += 'twelve'
			else:
				word += teens[o] + 'teen'
		else:
			word += tens[t] + 'ty '
	if o != 0:
		if t != 1:
			word += ones[o]
	count += len(word.replace(' ', ''))
count += len('onethousand')
print(count)