package xyz.laur.duels;

import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.plugin.Plugin;
import org.bukkit.scheduler.BukkitTask;

import java.io.IOException;
import java.net.URI;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class EmblemRenderer {
    private static final Random random = new Random();
    public static final Schematic[] EMBLEMS;

    private final Plugin plugin;
    private final List<Block> currentChanges = new ArrayList<>();
    private BukkitTask task;

    public EmblemRenderer(Plugin plugin) {
        this.plugin = plugin;
    }

    public void start() {
        task = plugin.getServer().getScheduler().runTaskTimer(plugin, this::changeEmblems, 0, 800);
    }

    public void stop() {
        task.cancel();
        removeExistingEmblems();
    }

    private void changeEmblems() {
        removeExistingEmblems();

        for (GameSession.GameMap map : GameSession.GameMap.values()) {
            for (EmblemPosition pos : randomEmblemPlacement(map)) {
                Schematic newEmblem = pickEmblem();
                if (newEmblem != null) {
                    showEmblem(pos, newEmblem);
                }
            }
        }
    }

    private void removeExistingEmblems() {
        for (Block block : currentChanges) {
            block.setType(Material.AIR);
        }
        currentChanges.clear();
    }

    private void showEmblem(EmblemPosition pos, Schematic emblem) {
        World world = getWorld();

        if (pos.isPlacedAlongXAxis()) {
            int startX = pos.getX() - emblem.getLength() / 2;
            int startY = pos.getY();
            int startZ = pos.getZ();

            for (int relX = 0; relX < emblem.getLength(); relX++) {
                for (int relY = 0; relY < emblem.getHeight(); relY++) {
                    Material type = emblem.getBlocks()[0][relY][relX];
                    byte data = emblem.getBlockDatas()[0][relY][relX];

                    Block block = world.getBlockAt(relX + startX, relY + startY, startZ);
                    if (block.getType() == Material.AIR) {
                        block.setType(type);
                        //noinspection deprecation
                        block.setData(data);

                        currentChanges.add(block);
                    }
                }
            }
        } else {
            int startX = pos.getX();
            int startY = pos.getY();
            int startZ = pos.getZ() - emblem.getLength() / 2;

            for (int relZ = 0; relZ < emblem.getLength(); relZ++) {
                for (int relY = 0; relY < emblem.getHeight(); relY++) {
                    Material type = emblem.getBlocks()[0][relY][relZ];
                    byte data = emblem.getBlockDatas()[0][relY][relZ];

                    Block block = world.getBlockAt(startX, relY + startY, relZ + startZ);
                    if (block.getType() == Material.AIR) {
                        block.setType(type);
                        //noinspection deprecation
                        block.setData(data);

                        currentChanges.add(block);
                    }
                }
            }
        }
    }

    private World getWorld() {
        List<World> worlds = plugin.getServer().getWorlds();
        assert worlds.size() == 1;
        return worlds.get(0);
    }

    static {
        try {
            EMBLEMS = getEmblems();

            validateEmblems();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static EmblemPosition[] randomEmblemPlacement(GameSession.GameMap map) {
        return map.getEmblemPositions()[random.nextInt(map.getEmblemPositions().length)];
    }

    private static Schematic pickEmblem() {
        if (random.nextDouble() < 0.4) {
            return EMBLEMS[random.nextInt(EMBLEMS.length)];
        } else {
            return null;
        }
    }

    private static Schematic[] getEmblems() throws Exception {
        URI directoryURI = EmblemRenderer.class.getClassLoader().getResource("emblems").toURI();

        try {
            Paths.get(directoryURI);
        } catch (FileSystemNotFoundException e) {
            FileSystems.newFileSystem(directoryURI, Collections.emptyMap());
        }

        return Files.list(Paths.get(directoryURI))
                .map(path -> {
                    try {
                        return Schematic.parse(Files.newInputStream(path));
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                })
                .toArray(Schematic[]::new);
    }

    private static void validateEmblems() {
        for (Schematic emblem : EMBLEMS) {
            assert emblem.getWidth() == 1;
        }
    }
}
