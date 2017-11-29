import pickle



def main():
    answers()


def answers():
    answer = pickle.load(open("A32078_Ficha1_Respostas.p", "rb"))

    # Ex 1 answers
    print("Ex1\n")

    # Point A
    answer["Q001"][0] = False
    # Point B
    answer["Q001"][1] = False
    # Point C
    answer["Q001"][2] = True
    # Point D
    answer["Q001"][3] = False
    # Point E
    answer["Q001"][4] = False
    # Point F
    answer["Q001"][5] = False

    print(answer["Q001"][0])
    print(answer["Q001"][1])
    print(answer["Q001"][2])
    print(answer["Q001"][3])
    print(answer["Q001"][4])
    print(answer["Q001"][5])

    # Ex 2 answers
    print("Ex2\n")

    # Point A
    answer["Q002"][0] = False
    # Point B
    answer["Q002"][1] = False
    # Point C
    answer["Q002"][2] = True
    # Point D
    answer["Q002"][3] = False
    # Point E True
    answer["Q002"][4] = False
    # Point F
    answer["Q002"][5] = False

    print(answer["Q002"][0])
    print(answer["Q002"][1])
    print(answer["Q002"][2])
    print(answer["Q002"][3])
    print(answer["Q002"][4])
    print(answer["Q002"][5])

    # Ex 3 answers

    print("Ex3\n")
    # Point A
    answer["Q003"][0] = False
    # Point B
    answer["Q003"][1] = True
    # Point C
    answer["Q003"][2] = False
    # Point D
    answer["Q003"][3] = False

    print(answer["Q003"][0])
    print(answer["Q003"][1])
    print(answer["Q003"][2])
    print(answer["Q003"][3])

    #Ex 4 answers

    print("Ex4\n")

    # Point A
    answer["Q004"][0] = False
    # Point B
    answer["Q004"][1] = False
    # Point C
    answer["Q004"][2] = False
    # Point D
    answer["Q004"][3] = False

    print(answer["Q004"][0])
    print(answer["Q004"][1])
    print(answer["Q004"][2])
    print(answer["Q004"][3])

    pickle.dump(answer, open("A32078_Ficha1_Respostas.p", "wb"))

main()
