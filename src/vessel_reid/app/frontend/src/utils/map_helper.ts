const EARTH_RADIUS = 6371;

/**
 * 
 * @param marker1 [Lat, Lon] of first vessel
 * @param marker2 [Lat, Lon] of second vessel
 * @returns Halfway point between the 2 vessels, to be used as the center point for the map
 */
export function getMapCentre(marker1: [number, number], marker2: [number, number]): [number, number] {
    return [(marker1[0] + marker2[0]) / 2, (marker1[1] + marker2[1]) / 2];
}

/**
 * 
 * @param marker1 [Lat, Lon] of first vessel
 * @param marker2 [Lat, Lon] of second vessel
 * @returns The level of zoom that should be used on the PigeonMaps map to display both points clearly
 */
export function getZoom(marker1: [number, number], marker2: [number, number]): number {
    const distance = Math.sqrt(
        Math.pow(marker1[0] - marker2[0], 2) +
        Math.pow(marker1[1] - marker2[1], 2)
    );
    return distance < 0.5 ? 10 : distance < 60 ? 4 : 2;
}

/**
 * https://en.wikipedia.org/wiki/Haversine_formula
 * The Pythagorean theorem is too simple because the Earth is round
 * @param marker1 [Lat, Lon] of first vessel
 * @param marker2 [Lat, Lon] of second vessel
 * @returns The Haversine distance between the 2 vessels
 */
export function getHaversineDistance(marker1: [number, number], marker2: [number, number]): number {
    const deltaLat = (marker2[0] - marker1[0]) * Math.PI / 180;
    const deltaLon = (marker2[1] - marker1[1]) * Math.PI / 180;
    const a = 
        Math.sin(deltaLat / 2) * Math.sin(deltaLat / 2) +
        Math.cos(marker1[0] * Math.PI / 180) * Math.cos(marker2[0] * Math.PI / 180) * Math.sin(deltaLon / 2) * Math.sin(deltaLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return EARTH_RADIUS * c;
}

/**
 * 
 * @param t1 Time 1
 * @param t2 Time 2
 * @returns Absolute value of the difference in hours between t1 and t2
 */
export function getTimeDifference(t1: string, t2: string): number {
    const ms = Math.abs(new Date(t1).getTime() - new Date(t2).getTime());
    return ms / (1000 * 60 * 60);
}

