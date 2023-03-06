from mytrain import train5n_SGD, train5s_SGD, train5n_AdamW, train5s_AdamW

if __name__=='__main__':
    # opt=train5n_SGD.parse_opt()
    # main=train5n_SGD.main(opt)
    # opt = train5n_AdamW.parse_opt()
    # main = train5n_AdamW.main(opt)
    # opt=train5s_SGD.parse_opt()
    # main=train5s_SGD.main(opt)
    opt = train5s_AdamW.parse_opt()
    main = train5s_AdamW.main(opt)
