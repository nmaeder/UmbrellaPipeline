def display_time(seconds: float) -> str:
    """
    Nice string representation for time
    Args:
        seconds (float): time you want a strin gfrom

    Returns:
        str: formatted time.
    """
    ret = ""
    tot = seconds
    intervals = [
        (604800, 0),
        (86400, 0),
        (3600, 0),
        (60, 0),
        (1, 0),
    ]
    for number, count in intervals:
        while tot >= number:
            count += 1
            tot -= number
        if count:
            ret += f"{count:02d}:"
    ret = ret.rstrip(":")
    return f"00:{ret}"
