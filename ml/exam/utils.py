
def _wrap_date(date):
    """
    transform datetime.datetime into datetime.date

    :type date: datetime.date | datetime.datetime
    :rtype: datetime.date
    """
    if isinstance(date, datetime.datetime):
        date = date.date()
    return date


def _validate_date(*dates):
    """
    check if the date(s) is supported

    :type date: datetime.date | datetime.datetime
    :rtype: datetime.date | list[datetime.date]
    """
    if len(dates) != 1:
        return list(map(_validate_date, dates))
    date = _wrap_date(dates[0])
    if not isinstance(date, datetime.date):
        raise NotImplementedError("unsupported type {}, expected type is datetime.date".format(type(date)))
    min_year, max_year = min(holidays.keys()).year, max(holidays.keys()).year
    if not (min_year <= date.year <= max_year):
        raise NotImplementedError(
            "no available data for year {}, only year between [{}, {}] supported".format(date.year, min_year, max_year)
        )
    return date

def is_workday(date):
    """
    check if one date is workday in China.
    in other words, Chinese people works at that day.

    :type date: datetime.date | datetime.datetime
    :rtype: bool
    """
    date = _validate_date(date)

    weekday = date.weekday()
    return bool(date in workdays.keys() or (weekday <= 4 and date not in holidays.keys()))
