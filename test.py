
games = 0
while games < 100:
    #env.run(agent)
    fd = open("test1.csv",'a')
    #to_write = str(games)
    fd.write("hello,world\n")
    games = games + 1