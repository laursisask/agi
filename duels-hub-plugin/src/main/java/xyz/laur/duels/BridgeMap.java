package xyz.laur.duels;

import org.bukkit.Location;

public enum BridgeMap {
    TREEHOUSE(
            "Treehouse",
            new Location(null, -28.5, 99, 0.5, -90, 0),
            new Location(null, 28.5, 99, 0.5, 90, 0),
            new Location(null, -33, 89, 0),
            new Location(null, 33, 89, 0),
            79
    ),
    ASHGATE(
            "Ashgate",
            new Location(null, -12.5, 75, 79.5, -90, 0),
            new Location(null, 43.5, 75, 79.5, 90, 0),
            new Location(null, -21, 62, 79),
            new Location(null, 51, 62, 79),
            58
    ),
    ATLANTIS(
            "Atlantis",
            new Location(null, -7.5, 90, 204.5, -90, 0),
            new Location(null, 42.5, 90, 204.5, 90, 0),
            new Location(null, -23, 87, 204),
            new Location(null, 57, 87, 204),
            79
    ),
    CHRONON(
            "Chronon",
            new Location(null, -14.5, 86, 314.5, -90, 0),
            new Location(null, 37.5, 86, 314.5, 90, 0),
            new Location(null, -23, 76, 314),
            new Location(null, 45, 76, 314),
            69
    ),
    CONDO(
            "Condo",
            new Location(null, -88.5, 116, 440.5, -90, 0),
            new Location(null, -20.5, 116, 440.5, 90, 0),
            new Location(null, -93, 106, 440),
            new Location(null, -17, 106, 440),
            101
    ),
    DOJO(
            "Dojo",
            new Location(null, -16.5, 82, -147.5, -90, 0),
            new Location(null, 41.5, 82, -147.5, 90, 0),
            new Location(null, -21, 70, -148),
            new Location(null, 45, 70, -148),
            68
    ),
    FORTRESS(
            "Fortress",
            new Location(null, -42.5, 88, -251.5, -90, 0),
            new Location(null, 8.5, 88, -251.5, 90, 0),
            new Location(null, -50, 78, -252),
            new Location(null, 14, 78, -252),
            71
    ),
    GALAXY(
            "Galaxy",
            new Location(null, -24.5, 76, -412.5, -90, 0),
            new Location(null, 37.5, 76, -412.5, 90, 0),
            new Location(null, -29, 61, -413),
            new Location(null, 41, 61, -413),
            58
    ),
    LICORICE(
            "Licorice",
            new Location(null, -9.5, 30, -548.5, -90, 0),
            new Location(null, 56.5, 30, -548.5, 90, 0),
            new Location(null, -12, 20, -549),
            new Location(null, 58, 20, -549),
            15
    ),
    LIGHTHOUSE_V2(
            "Lighthouse V2",
            new Location(null, -208.5, 61, -43.5, -90, 0),
            new Location(null, -152.5, 61, -43.5, 90, 0),
            new Location(null, -215, 50, -44),
            new Location(null, -147, 50, -44),
            46
    ),
    MISTER_CHEESY(
            "Mister Cheesy",
            new Location(null, -265.5, 98, 52.5, -90, 0),
            new Location(null, -217.5, 98, 52.5, 90, 0),
            new Location(null, -269, 89, 52),
            new Location(null, -215, 89, 52),
            84
    ),
    OUTPOST(
            "Outpost",
            new Location(null, -284.5, 120, 203.5, -90, 0),
            new Location(null, -230.5, 120, 203.5, 90, 0),
            new Location(null, -292, 108, 203),
            new Location(null, -224, 108, 203),
            104
    ),
    PALAESTRA(
            "Palaestra",
            new Location(null, -288.5, 80, 392.5, -90, 0),
            new Location(null, -226.5, 80, 392.5, 90, 0),
            new Location(null, -292, 66, 392),
            new Location(null, -224, 66, 392),
            61
    ),
    STUMPED(
            "Stumped",
            new Location(null, -186.5, 54, -172.5, -90, 0),
            new Location(null, -134.5, 54, -172.5, 90, 0),
            new Location(null, -195, 49, -173),
            new Location(null, -127, 49, -173),
            44
    ),
    SUNSTONE(
            "Sunstone",
            new Location(null, -177.5, 88, -303.5, -90, 0),
            new Location(null, -129.5, 88, -303.5, 90, 0),
            new Location(null, -184, 77, -304),
            new Location(null, -124, 77, -304),
            71
    ),
    TUNDRA_V2(
            "Tundra V2",
            new Location(null, -214.5, 48, -429.5, -90, 0),
            new Location(null, -163.5, 48, -429.5, 90, 0),
            new Location(null, -218, 36, -430),
            new Location(null, -162, 36, -430),
            31
    ),
    URBAN(
            "Urban",
            new Location(null, 201.5, 90, -20.5, -90, 0),
            new Location(null, 265.5, 90, -20.5, 90, 0),
            new Location(null, 200, 80, -21),
            new Location(null, 266, 80, -21),
            77
    );

    private final String displayName;
    private final Location blueSpawn;
    private final Location redSpawn;
    private final Location blueHole;
    private final Location redHole;
    private final int minY;

    BridgeMap(String displayName, Location blueSpawn, Location redSpawn, Location blueHole, Location redHole, int minY) {
        this.displayName = displayName;
        this.blueSpawn = blueSpawn;
        this.redSpawn = redSpawn;
        this.blueHole = blueHole;
        this.redHole = redHole;
        this.minY = minY;
    }

    public String getDisplayName() {
        return displayName;
    }

    public Location getBlueSpawn() {
        return blueSpawn;
    }

    public Location getRedSpawn() {
        return redSpawn;
    }

    public Location getBlueHole() {
        return blueHole;
    }

    public Location getRedHole() {
        return redHole;
    }

    public int getMinY() {
        return minY;
    }

    public static BridgeMap fromName(String name) {
        for (BridgeMap map : values()) {
            if (map.name().equalsIgnoreCase(name)) {
                return map;
            }
        }

        throw new IllegalArgumentException("No map with name " + name);
    }
}
