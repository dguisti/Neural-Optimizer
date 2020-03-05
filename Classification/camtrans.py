with open("trashcode.py", 'r') as f:
    cameronscode = f.readlines();
garbagecode = cameronscode;

with open("trashcode.py", 'w') as trash:
    for line in garbagecode:
        char = line[0]
        charnum = 0;

        while char == " ":
            char = line[charnum]
            charnum += 1;
        
        charnum = int(charnum/4)
        
        newline = ''.join(['   ' for sp in range(charnum)]) + line.replace(' ', '').replace(',', ', ')
        print(newline, file=trash, end='')