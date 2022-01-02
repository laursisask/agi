package xyz.laur.duels;

public class EmblemPosition {
    private final int x;
    private final int y;
    private final int z;
    private final boolean placedAlongXAxis;

    public EmblemPosition(int x, int y, int z, boolean placedAlongXAxis) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.placedAlongXAxis = placedAlongXAxis;
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }

    public int getZ() {
        return z;
    }

    public boolean isPlacedAlongXAxis() {
        return placedAlongXAxis;
    }
}
