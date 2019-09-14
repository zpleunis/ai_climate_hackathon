import datetime
import matplotlib.pyplot as plt
import numpy as np
import pytz
import scipy.stats

WS_DTYPE = [("x", np.float),
            ("y", np.float),
            ("station_name", np.str_, 64),
            ("stn_id", np.int),
            ("climate_identifier", np.int),
            ("id", np.str_, 64),
            ("local_date", "datetime64[ms]"),
            ("province_code", np.str_, 2),
            ("local_year", np.float),
            ("local_month", np.float),
            ("local_day", np.float),
            ("mean_temperature", np.float),
            ("mean_temperature_flag", np.str_, 1),
            ("min_temperature", np.float),
            ("min_temperature_flag", np.str_, 1),
            ("max_temperature", np.float),
            ("max_temperature_flag", np.str_, 1),
            ("total_precipitation", np.float),
            ("total_precipitation_flag", np.str_, 1),
            ("total_rain", np.float),
            ("total_rain_flag", np.str_, 1),
            ("total_snow", np.float),
            ("total_snow_flag", np.str_, 1),
            ("snow_on_ground", np.float),
            ("snow_on_ground_flag", np.str_, 1),
            ("direction_max_gust", np.float),
            ("direction_max_gust_flag", np.str_, 1),
            ("speed_max_gust", np.float),
            ("speed_max_gust_flag", np.str_, 1),
            ("cooling_degree_days", np.float),
            ("cooling_degree_days_flag", np.str_, 1),
            ("heating_degree_days", np.float),
            ("heating_degree_days_flag", np.str_, 1),
            ("min_rel_humidity", np.float),
            ("min_rel_humidity_flag", np.str_, 1),
            ("max_rel_humidity", np.float),
            ("max_rel_humidity_flag", np.str_, 1)]

VARS = ["min_temperature", "mean_temperature", "max_temperature",
        "total_precipitation", "total_rain", "total_snow",
        "snow_on_ground"]


def angular_separation(ra1, dec1, ra2, dec2):
    """Calculates the angular separation between two celestial objects.

    Parameters
    ----------
    ra1 : float, array
        Right ascension of the first source in degrees.
    dec1 : float, array
        Declination of the first source in degrees.
    ra2 : float, array
        Right ascension of the second source in degrees.
    dec2 : float, array
        Declination of the second source in degrees.

    Returns
    -------
    angle : float, array
        Angle between the two sources in degrees, where 0 corresponds
        to the positive Dec axis.
    angular_separation : float, array
        Angular separation between the two sources in degrees.

    Notes
    -----
    The angle between sources is calculated with the Pythagorean theorem
    and is used later to calculate the uncertainty in the event position
    in the direction of the known source.

    Calculating the angular separation using spherical geometry gives
    poor accuracy for small (< 1 degree) separations, and using the
    Pythagorean theorem fails for large separations (> 1 degrees).
    Transforming the spherical geometry cosine formula into one that
    uses haversines gives the best results, see e.g. [1]_. This gives:

    .. math:: \mathrm{hav} d = \mathrm{hav} \Delta\delta +
              \cos\delta_1 \cos\delta_2 \mathrm{hav} \Delta\\alpha

    Where we use the identity
    :math:`\mathrm{hav} \\theta = \sin^2(\\theta/2)` in our
    calculations.

    The calculation might give inaccurate results for antipodal
    coordinates, but we do not expect such calculations here..

    The source angle (or bearing angle) :math:`\\theta` from a point
    A(ra1, dec1) to a point B(ra2, dec2), defined as the angle in
    clockwise direction from the positive declination axis can be
    calculated using:

    .. math:: \\tan(\\theta) = (\\alpha_2 - \\alpha_1) /
              (\delta_2 - \delta_1)

    In NumPy :math:`\\theta` can be calculated using the arctan2
    function. Note that for negative :math:`\\theta` a factor :math:`2\pi`
    needs to be added. See also the documentation for arctan2.

        References
    ----------
    .. [1] Sinnott, R. W. 1984, Sky and Telescope, 68, 158

    Examples
    --------
    >>> print(angular_separation(200.478971, 55.185900, 200.806433, 55.247994))
    (79.262937451490941, 0.19685681276638525)
    >>> print(angular_separation(0., 20., 180., 20.))
    (90.0, 140.0)

    """
    # convert decimal degrees to radians
    deg2rad = np.pi / 180
    ra1 = ra1 * deg2rad
    dec1 = dec1 * deg2rad
    ra2 = ra2 * deg2rad
    dec2 = dec2 * deg2rad

    # delta works
    dra = ra1 - ra2
    ddec = dec1 - dec2

    # haversine formula
    hav = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) \
        * np.sin(dra / 2.0) ** 2
    angular_separation = 2 * np.arcsin(np.sqrt(hav))

    # angle in the clockwise direction from the positive dec axis
    # note the minus signs in front of `dra` and `ddec`
    source_angle = np.arctan2(-dra, -ddec)
    source_angle[source_angle < 0] += 2 * np.pi

    # convert radians back to decimal degrees
    return source_angle / deg2rad, angular_separation / deg2rad


def calculate_moments(ts):
    """Calculate the moments of a timeseries."""
    mean = np.nanmean(ts)
    sigma = np.nanstd(ts)
    skew = scipy.stats.skew(ts, nan_policy="omit")
    kurt = scipy.stats.kurtosis(ts, nan_policy="omit")
    return mean, sigma, skew, kurt


class WSData():
    """Canadian weather station data."""
    def __init__(self, fname):
        self.data = np.genfromtxt(fname, dtype=WS_DTYPE,
                                  delimiter=",", skip_header=1)

        self.unique_stations, self.unique_stations_idx = \
            np.unique(self.data["stn_id"], return_index=True)

    def find_nearest_station(self, lon, lat):
        """Find the station ID of the weather station nearest to some coordinate."""
        angle, sep = angular_separation(
            self.data[self.unique_stations_idx]["x"],
            self.data[self.unique_stations_idx]["y"], lon, lat)

        nearest_idx = np.argmin(sep)

        print("")
        print("Nearest weather station is {:.2f} degrees away..".format(
            sep[nearest_idx]))
        print("")

        return self.unique_stations[nearest_idx]

    def retrieve_vector(self, lon, lat, date):
        """Retrieve data for a given riding and election.

        Parameters
        ----------
        lon : float
            Longitude, in degrees.
        lat : float
            Latitude, in degrees.
        date : :obj:datetime
            Date.

        Returns
        -------
        vector : array_like
            Data vector for model.

        """
        station = self.find_nearest_station(lon, lat)

        # no of variables x 4 moments x 8 time ranges
        vector = np.empty(len(VARS) * (2 * 1 + 4 * 6))
        i = 0

        station_data = self.data[self.data["stn_id"] == station]

        # TODO pick the variable to keep
        # TODO interpolate to replace nans
        for var in VARS:

            for timerange in [0, -1, -7, -30, -365, -730, -1095, -1460]:

                if timerange == 0:
                    # election day
                    mask = station_data["local_date"] == date
                    station_time_data = station_data[mask]
                elif timerange == -1:
                    # day before election day
                    mask = station_data["local_date"] \
                        == date + datetime.timedelta(days=timerange)
                    station_time_data = station_data[mask]
                elif timerange > -366:
                    # week, month or year before election day
                    mask = np.logical_and(station_data["local_date"] <= date,
                                          station_data["local_date"] > date \
                                          + datetime.timedelta(days=timerange))
                    station_time_data = station_data[mask]
                elif timerange == -730:
                    # 2 years before election day
                    mask = np.logical_and(station_data["local_date"] <= date \
                                          + datetime.timedelta(days=-365),
                                          station_data["local_date"] > date \
                                          + datetime.timedelta(days=timerange))
                    station_time_data = station_data[mask]
                elif timerange == -1095:
                    # 3 years before election day
                    mask = np.logical_and(station_data["local_date"] <= date \
                                          + datetime.timedelta(days=-730),
                                          station_data["local_date"] > date \
                                          + datetime.timedelta(days=timerange))
                    station_time_data = station_data[mask]
                else:
                    # 4 years before election day
                    mask = np.logical_and(station_data["local_date"] <= date \
                                          + datetime.timedelta(days=-1095),
                                          station_data["local_date"] > date \
                                          + datetime.timedelta(days=timerange))
                    station_time_data = station_data[mask]

                ts = station_time_data[var]

                if len(ts) == 1:
                    vector[i] = ts[0]
                    i += 1
                else:
                    moments = calculate_moments(ts)
                    vector[i:i+4] = moments
                    i += 4

        # TODO add direction max gust

        print("vector:", vector)
        print("vector.shape:", vector.shape)

        return vector


if __name__  == "__main__":

    lon = -70.
    lat = 44.
    election_date = datetime.datetime(2019, 4, 30)
    weather_data = WSData("weather_montreal.csv")
    vector = weather_data.retrieve_vector(lon, lat, election_date)
