package xyz.laur.duels;

import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.Plugin;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SumoGameSession implements Listener, GameSession {
    public static final long MAX_DURATION = 6000; // 5 minutes

    public enum GameMap {
        CLASSIC_SUMO(
                "Classic Sumo",
                new Location[][]{
                        new Location[]{new Location(null, 127.5, 44, 93.5, -90, 0), new Location(null, 139.5, 44, 93.5, 90, 0)},
                        new Location[]{new Location(null, 133.5, 44, 99, -180, 0), new Location(null, 133.5, 44, 87.5, 0, 0)}
                },
                new EmblemPosition[][]{
                        new EmblemPosition[]{new EmblemPosition(124, 49, 93, false), new EmblemPosition(142, 49, 93, false)},
                        new EmblemPosition[]{new EmblemPosition(133, 49, 102, true), new EmblemPosition(133, 49, 84, true)}
                },
                41
        ),
        SPACE_MINE(
                "Space Mine",
                new Location[][]{
                        new Location[]{new Location(null, 161.5, 144, 54.5, -90, 0), new Location(null, 173.5, 144, 54.5, 90, 0)},
                        new Location[]{new Location(null, 167.5, 144, 60.5, -180, 0), new Location(null, 167.5, 144, 48.5, 0, 0)}
                },
                new EmblemPosition[][]{
                        new EmblemPosition[]{new EmblemPosition(176, 149, 54, false), new EmblemPosition(158, 149, 54, false)},
                        new EmblemPosition[]{new EmblemPosition(167, 149, 45, true), new EmblemPosition(167, 149, 63, true)}
                },
                141
        ),
        WHITE_CRYSTAL(
                "White Crystal",
                new Location[][]{
                        new Location[]{new Location(null, 214.5, 46, 169.5, -90, 0), new Location(null, 226.5, 46, 169.5, 90, 0)},
                        new Location[]{new Location(null, 220.5, 46, 175.5, 180, 0), new Location(null, 220.5, 46, 163.5, 0, 0)}
                },
                new EmblemPosition[][]{
                        new EmblemPosition[]{new EmblemPosition(211, 51, 169, false), new EmblemPosition(229, 51, 169, false)},
                        new EmblemPosition[]{new EmblemPosition(220, 51, 178, true), new EmblemPosition(220, 51, 160, true)}
                },
                43
        ),
        PONSEN(
                "Ponsen",
                new Location[][]{
                        new Location[]{new Location(null, 216.5, 33, 29.5, -90, 0), new Location(null, 228.5, 33, 29.5, 90, 0)},
                        new Location[]{new Location(null, 222.5, 33, 35.5, 180, 0), new Location(null, 222.5, 33, 23.5, 0, 0)}
                },
                new EmblemPosition[][]{
                        new EmblemPosition[]{new EmblemPosition(213, 38, 29, false), new EmblemPosition(231, 38, 29, false)},
                        new EmblemPosition[]{new EmblemPosition(222, 38, 38, true), new EmblemPosition(222, 38, 20, true)}
                },
                30
        ),
        FORT_ROYALE(
                "Fort Royale",
                new Location[][]{
                        new Location[]{new Location(null, 119.5, 44, 12.5, 90, 0), new Location(null, 107.5, 44, 12.5, -90, 0)},
                        new Location[]{new Location(null, 113.5, 44, 6.5, 0, 0), new Location(null, 113.5, 44, 18.5, 180, 0)}
                },
                new EmblemPosition[][]{
                        new EmblemPosition[]{new EmblemPosition(122, 49, 12, false), new EmblemPosition(104, 49, 12, false)},
                        new EmblemPosition[]{new EmblemPosition(113, 49, 3, true), new EmblemPosition(113, 49, 21, true)}
                },
                41
        );

        private final String displayName;
        private final Location[][] spawnLocations;
        private final EmblemPosition[][] emblemPositions;
        private final double minY;

        GameMap(String displayName, Location[][] spawnLocations, EmblemPosition[][] emblemPositions, double minY) {
            this.displayName = displayName;
            this.spawnLocations = spawnLocations;
            this.emblemPositions = emblemPositions;
            this.minY = minY;
        }

        public static GameMap fromName(String name) {
            for (GameMap map : values()) {
                if (map.name().equalsIgnoreCase(name)) {
                    return map;
                }
            }

            throw new IllegalArgumentException("No map with name " + name);
        }

        public String getDisplayName() {
            return displayName;
        }

        public EmblemPosition[][] getEmblemPositions() {
            return emblemPositions;
        }
    }

    private GameState state = GameState.WAITING_FOR_PLAYERS;
    private final List<Player> players = new ArrayList<>();
    private final GameMap map;
    private final boolean randomTeleport;
    private final SessionManager sessionManager;
    private final InvisibilityManager invisibilityManager;
    private final SkinChanger skinChanger;
    private final Plugin plugin;
    private final Location[] spawnLocations;
    private static final Random random = new Random();

    private int randomTeleportTask;

    public SumoGameSession(World world, SessionManager sessionManager, InvisibilityManager invisibilityManager,
                           SkinChanger skinChanger, Plugin plugin, float randomizationFactor, GameMap map,
                           boolean randomTeleport) {
        this.sessionManager = sessionManager;
        this.invisibilityManager = invisibilityManager;
        this.skinChanger = skinChanger;
        this.plugin = plugin;
        this.map = map;
        this.randomTeleport = randomTeleport;

        int spawnLocPair = pickSpawnLocationPair(randomizationFactor);

        spawnLocations = new Location[2];
        for (int i = 0; i < map.spawnLocations.length; i++) {
            spawnLocations[i] = map.spawnLocations[spawnLocPair][i].clone();
            spawnLocations[i].setWorld(world);
        }

        plugin.getServer().getPluginManager().registerEvents(this, plugin);

        plugin.getServer().getScheduler().scheduleSyncDelayedTask(plugin, () -> {
            if (state == GameState.PLAYING) {
                endGame(null);
                plugin.getLogger().info("Game ended due to timeout");
            }
        }, MAX_DURATION);
    }

    public GameState getState() {
        return state;
    }

    public GameMap getMap() {
        return map;
    }

    @Override
    public String getMapName() {
        return map.getDisplayName();
    }

    public boolean hasRandomTeleport() {
        return randomTeleport;
    }

    public List<Player> getPlayers() {
        return players;
    }

    public void addPlayer(Player player) {
        if (state != GameState.WAITING_FOR_PLAYERS) throw new IllegalStateException("Session is in invalid state");
        if (players.size() >= 2) throw new IllegalStateException("Game already has too many players");
        if (hasPlayer(player)) throw new IllegalArgumentException("Player is already in this session");
        if (sessionManager.getPlayerSession(player) != null)
            throw new IllegalArgumentException("Player is already in a session");

        players.add(player);
        invisibilityManager.update();

        if (players.size() >= 2) {
            startGame();
        }
    }

    public void startGame() {
        if (state != GameState.WAITING_FOR_PLAYERS) throw new IllegalStateException("Session is in invalid state");
        if (players.size() != 2) throw new IllegalStateException("Need exactly 2 players to start a game");

        plugin.getLogger().info("Starting sumo duel game");

        for (int i = 0; i < players.size(); i++) {
            Player player = players.get(i);
            skinChanger.changeSkin(player);

            player.teleport(spawnLocations[i]);

            player.sendMessage("Game started");
            player.setHealth(player.getMaxHealth());
            player.getInventory().clear();
        }

        if (randomTeleport) {
            randomTeleportTask = plugin.getServer().getScheduler().scheduleSyncRepeatingTask(plugin,
                    this::randomTeleportTick, 40, 1);
        }

        state = GameState.PLAYING;
    }

    public void endGame(Player winner) {
        if (state == GameState.ENDED) throw new IllegalStateException("Game has already ended");

        state = GameState.ENDED;

        PlayerQuitEvent.getHandlerList().unregister(this);
        PlayerMoveEvent.getHandlerList().unregister(this);
        EntityDamageByEntityEvent.getHandlerList().unregister(this);
        EntityDamageEvent.getHandlerList().unregister(this);

        if (winner == null) {
            for (Player player : players) {
                player.sendMessage("You lost");
            }
        } else {
            sendMetadata(winner, "win", 1);
            winner.sendMessage("You won");

            Player loser = getOtherPlayer(winner);
            loser.sendMessage("You lost");
        }

        sessionManager.removeSession(this);
        invisibilityManager.update();

        if (randomTeleport) {
            plugin.getServer().getScheduler().cancelTask(randomTeleportTask);
        }
    }

    public boolean hasPlayer(Player player) {
        return players.contains(player);
    }

    protected static int pickSpawnLocationPair(float randomizationFactor) {
        // when randomizationFactor 0 - always pick the first pair
        // when randomizationFactor 1 - pick completely randomly
        if (random.nextDouble() < randomizationFactor) {
            return random.nextInt(2);
        }

        return 0;
    }

    protected void randomTeleportTick() {
        if (state != GameState.PLAYING) {
            throw new IllegalStateException("Game was not in playing state but random teleport tick was called");
        }

        if (!randomTeleport) {
            throw new IllegalStateException("Game did not have random teleport in but random teleport tick was called");
        }

        Player player = randomPlayer();

        double distanceBetweenPlayers = player.getLocation().distance(getOtherPlayer(player).getLocation());
        if (distanceBetweenPlayers < 3 && random.nextDouble() < 1D / 100) {
            plugin.getLogger().info("Randomly teleporting one player");
            Location newLocation = player.getLocation().clone();
            newLocation.add(4 * (random.nextDouble() - 0.5), 0, 4 * (random.nextDouble() - 0.5));
            newLocation.setPitch(newLocation.getPitch() + (random.nextFloat() - 0.5F) * 30);
            newLocation.setYaw(newLocation.getYaw() + (random.nextFloat() - 0.5F) * 30);
            player.teleport(newLocation);
        }
    }

    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        if (!hasPlayer(event.getPlayer()) || state != GameState.PLAYING) return;

        Player winner = getOtherPlayer(event.getPlayer());
        endGame(winner);

        plugin.getLogger().info("Sumo game ended due to one player quiting");
    }

    @EventHandler
    public void onMove(PlayerMoveEvent event) {
        Player player = event.getPlayer();
        if (!hasPlayer(player) || state != GameState.PLAYING) return;

        if (event.getTo().getY() < map.minY) {
            Player winner = getOtherPlayer(player);
            endGame(winner);
            plugin.getLogger().info("Sumo game ended due to one player falling off the platform");
        }
    }

    @EventHandler
    public void onDamage(EntityDamageEvent event) {
        if (event.getEntity() instanceof Player && hasPlayer((Player) event.getEntity())) {
            event.setDamage(0);
        }
    }

    @EventHandler
    public void onPlayerDamage(EntityDamageByEntityEvent event) {
        if (!(event.getEntity() instanceof Player) || !(event.getDamager() instanceof Player) ||
                state != GameState.PLAYING) {
            return;
        }

        Player attacker = (Player) event.getDamager();
        Player target = (Player) event.getEntity();

        if (hasPlayer(attacker) && hasPlayer(target)) {
            sendMetadata(attacker, "hits_done", 1);
            sendMetadata(target, "hits_received", 1);
        }
    }

    private Player randomPlayer() {
        return players.get(random.nextInt(players.size()));
    }

    protected void sendMetadata(Player player, String key, double value) {
        player.sendMessage(String.format("metadata:%s:%.5f", key, value));
    }

    protected Player getOtherPlayer(Player player) {
        for (Player p : players) {
            if (!p.equals(player)) {
                return p;
            }
        }

        throw new IllegalStateException("Could not find other player");
    }
}
