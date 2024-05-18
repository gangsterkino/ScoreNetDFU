def _internal_calculate_bias(score, total_area, x1, x2, x3):
    bias =0
    if total_area <= 500 and score < 35:
        score = min(max(18, score), 25)
        bias = 7+(score / 23)
        print("ll")
    elif total_area <= 2000 and score > 50:
        score = min(max(55, score), 60)
        bias = 5 + (score / 13)

    elif total_area > 500 and total_area < 2000:
        if x1 >= (score - 5) and score > 30:
            bias = 8 + (score / 43)

        elif x3 > x2:
            if score<20 and total_area<1500:
                score = min(max(26, score), 35)
                bias = 4 + (score / 60)
                print("here")

            else:
                score =min(max(53,score),62)
                bias=(score/30)

        elif x2 > x3:
            score = min(max(46, score), 55)
            bias = 8 + (score / 33)

        elif total_area > 1700 and x1 >= (score - x2 - x3):
            score = min(max(53, score), 59)
            bias = 5 + (score / 37)
        elif total_area>6000:
            if x1>score/2:
                score=min(max(82,score),85)
                bias=3+score/11
            elif x1>score/3:
                score = min(max(49, score), 53)
                bias = 3 + score / 11
            else:
                score = min(max(57, score), 64)
                bias = 5 + (score / 20)

        else:
            score = min(max(25, score), 35)
            bias = (score / 60)

    elif total_area > 2000 and total_area < 11000:
        if total_area > 3000 or x1 >= (score - 1):
            if x3 > x1 and x3 > (score - x2 - x1):
                if x3 < x2:
                    if total_area>7000:
                        score=min(max(55,score),60)
                        bias=(score/27)
                    else:
                        score = min(max(53, score), 62)
                        bias = (score / 73)

                elif total_area > 4000:
                    score = min(max(50, score), 58)
                    bias = (score / 63)
                    print("b")

                else:
                    score = min(max(30, score), 35)
                    bias = 10 + (score / 23)

            elif x2 > x1 and x2 >= (score - x1 - x3):
                if x2 >= (score / 2):
                    if total_area > 4000:
                        score = min(max(53, score), 58)
                        bias = 4 + (score / 27)
                        print("A")
                    else:
                        score = min(max(38, score), 45)
                        bias = (score / 27)

                else:
                    score = min(max(43, score), 50)
                    bias = 5 + (score / 30)

            elif total_area > 4000:
                score = min(max(58, score), 63)
                bias = (score / 30)

            elif x3 > x2 and x3 >= (score - x1 - x2):
                if total_area > 5000:
                    score = min(max(49, score), 53)
                    bias = (score / 23)

                else:
                    score = min(max(48, score), 57)
                    bias = 3+(score / 23)
                    print("bk")
            elif x2 > x3 and x2 >= (score - x1 - x3):
                score = min(max(53, score), 65)
                bias = 4 + (score / 33)
            else:
                score = min(max(73, score), 78)
                bias = 4+ (score / 30)


        elif x1 < 30 and x1 >= (score / 6):
            score = min(max(54,score), 62)
            bias = 5 + (score / 23)
            print("6")
        elif x2 >= (score / 3):
            if total_area > 5000:
                if total_area>9000:
                    score = min(max(67, score), 75)
                    bias = (score / 40)
                else:
                    score = min(max(50, score), 55)
                    bias = (score / 40)
                    print("C")
            else:
                score = min(max(33, score), 42)
                bias = (score / 53)
        else:
            score = min(max(65, score), 75)
            bias = 5 + (score / 60)
    elif total_area > 11000 or x1 > 30:
        if total_area > 20000 and x1 > 10:
            score = 100
        elif total_area > 11000 and x1 < 10:
            if total_area>16000 and x3>x2:
                score = min(max(89, score), 93)
                bias = (score / 37)
            elif total_area>16000 and x1>(score-x1):
                score=min(max(92,score),100)
                bias =score/47
            elif total_area>30000:
                score=min(max(88,score),90)
                bias=score/43
            else:
                score = min(max(67, score), 73)
                bias = (score / 37)

        elif x1 >= (score - 1) or ((x2 + x3) >= (score - x1)):
            score = min(max(70, score), 74)
            bias = (score / 53)
            print("C")

        else:
            score = min(max(90, score), 100)
            bias =  (score / 20)
    else:
        bias = 10+(score / 23)

    return score,bias
