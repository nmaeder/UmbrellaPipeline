def display_time(seconds: float) -> str:
    """
    Converts float type seconds into a nice string representation

    Args:
        seconds (float): time you want a strin gfrom

    Returns:
        str: formatted time.
    """

    ret = ""
    sign = ""
    if seconds < 0:
        tot = abs(seconds)
        sign = "-"
    else:
        tot = seconds
    intervals = [
        604800,
        86400,
        3600,
        60,
        1,
    ]
    for inter in intervals:
        count = 0
        while tot >= inter:
            count += 1
            tot -= inter
        if count:
            ret += f"{count:02d}:"
        if inter == 1 and count == 0:
            ret += f"{count:02d}:"
    ret = ret.rstrip(":")
    return f"{sign}00:{ret}"
