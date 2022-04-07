package xyz.laur.duels;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.util.Vector;

import java.util.ArrayList;
import java.util.List;

public class MapBarrier {
    private final JavaPlugin plugin;
    private final ClassicGameSession.GameMap map;
    private final World world;
    private int mapSize;
    private List<Block> currentChanges = new ArrayList<>();

    public MapBarrier(JavaPlugin plugin, ClassicGameSession.GameMap map, World world) {
        this.plugin = plugin;
        this.map = map;
        this.world = world;
    }

    public void update(float spawnDistance) {
        Vector center = map.getSpawnLocation1().toVector().add(map.getSpawnLocation2().toVector()).multiply(0.5);
        int initialDistanceFromCenter = (int) map.getSpawnLocation1().toVector().distance(center);

        int newMapSize = (int) (3 + spawnDistance * initialDistanceFromCenter);

        if (newMapSize != mapSize) {
            removeBarrier();
            mapSize = newMapSize;
            applyBarrier();
        }
    }

    public void removeBarrier() {
        for (Block block : currentChanges) {
            block.setType(Material.AIR);
        }
        currentChanges.clear();
    }

    private void applyBarrier() {
        plugin.getLogger().info("Applying barrier of size " + mapSize + " to map " + map.getDisplayName());

        Location center = map.getSpawnLocation1()
                .toVector()
                .add(map.getSpawnLocation2().toVector())
                .multiply(0.5)
                .toLocation(world);

        int minY = Math.max(map.getSpawnLocation1().getBlockY() - 5, 0);
        int maxY = map.getSpawnLocation1().getBlockY() + 10;

        for (int y = minY; y < maxY; y++) {
            for (int i = -mapSize; i < mapSize; i++) {
                changeBlock(world.getBlockAt(center.getBlockX() + i, y, center.getBlockZ() - mapSize));
                changeBlock(world.getBlockAt(center.getBlockX() + i, y, center.getBlockZ() + mapSize));

                changeBlock(world.getBlockAt(center.getBlockX() - mapSize, y, center.getBlockZ() + i));
                changeBlock(world.getBlockAt(center.getBlockX() + mapSize, y, center.getBlockZ() + i));
            }
        }
    }

    private void changeBlock(Block block) {
        if (block.getType() == Material.AIR) {
            block.setType(Material.BARRIER);
            currentChanges.add(block);
        }
    }

}
