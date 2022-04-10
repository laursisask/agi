def linear_increase(iteration, increase_end):
    if iteration >= increase_end:
        return 1

    return 1 - (increase_end - iteration) / increase_end


def liner_anneal(iteration, decrease_end):
    if iteration >= decrease_end:
        return 0

    return (decrease_end - iteration) / decrease_end
