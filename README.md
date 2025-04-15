just after nfft = 2048
segmenting = 0  # 0 => the code will take one piece of data per file
                # 1 => the code will take len(data)/length segments per file (padding with zeros on last segment)
                # 2 => the code will take len(data)/length segments per file (last segment will end before end of file data when possible)

replace "sections = int(etc etc)
        if segmenting == 0:
            sections = int(1)
        elif segmenting == 1:
            sections = int(np.ceil(sourceLength/(length*sr)))
        elif segmenting == 2:
            sections = int(np.floor(sourceLength/(length*sr)))
        else:
            print("Bad input variable (segmenting != [0,1,2]")
