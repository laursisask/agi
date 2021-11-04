package xyz.laur.duels;

import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.Plugin;

import java.util.*;

public class GameSession implements Listener {
    public static final long MAX_DURATION = 6000; // 5 minutes
    public static final double HIT_REWARD = 5;

    public enum State {
        WAITING_FOR_PLAYERS, PLAYING, ENDED
    }

    public enum GameMap {
        CLASSIC_SUMO(
            "Classic Sumo",
            new Location[][]{
                new Location[]{new Location(null, -18, 77, 33, 90, 0), new Location(null, -30, 77, 33, -90, 0)},
                new Location[]{new Location(null, -24, 77, 39, 180, 0), new Location(null, -24, 77, 27, 0, 0)}
            },
            73,
            new Location(null, -24, 76, 33)
        ),
        SPACE_MINE(
            "Space Mine",
            new Location[][]{
                new Location[]{new Location(null, 101, 98, -189, 90, 0), new Location(null, 89, 98, -189, -90, 0)},
                new Location[]{new Location(null, 95, 98, -183, -180, 0), new Location(null, 95, 98, -195, 0, 0)}
            },
            93,
            new Location(null, 95, 96, -189)
        ),
        WHITE_CRYSTAL(
            "White Crystal",
            new Location[][]{
                new Location[]{new Location(null, 179, 80, -22, -90, 0), new Location(null, 191, 80, -22, 90, 0)},
                new Location[]{new Location(null, 185, 80, -16, 180, 0), new Location(null, 185, 80, -28, 0, 0)}
            },
            74,
            new Location(null, 185, 78, -22)
        );

        private final String displayName;
        private final Location[][] spawnLocations;
        private final double minY;
        private final Location center;

        GameMap(String displayName, Location[][] spawnLocations, double minY, Location center) {
            this.displayName = displayName;
            this.spawnLocations = spawnLocations;
            this.minY = minY;
            this.center = center;
        }

        public String getDisplayName() {
            return displayName;
        }
    }

    private State state = State.WAITING_FOR_PLAYERS;
    private final List<Player> players = new ArrayList<>();
    private final GameMap map;
    private final SessionManager sessionManager;
    private final InvisibilityManager invisibilityManager;
    private final Plugin plugin;
    private final Location[] spawnLocations;
    private final Map<Player, Location> lastLocations = new HashMap<>();
    private static final Random random = new Random();

    public GameSession(World world, SessionManager sessionManager, InvisibilityManager invisibilityManager,
                       Plugin plugin, float randomizationFactor) {
        this.sessionManager = sessionManager;
        this.invisibilityManager = invisibilityManager;
        this.plugin = plugin;

        map = pickMap(randomizationFactor);

        int spawnLocPair = pickSpawnLocationPair(randomizationFactor);

        spawnLocations = new Location[2];
        for (int i = 0; i < map.spawnLocations.length; i++) {
            spawnLocations[i] = map.spawnLocations[spawnLocPair][i].clone();
            spawnLocations[i].setWorld(world);
        }

        plugin.getServer().getPluginManager().registerEvents(this, plugin);

        plugin.getServer().getScheduler().scheduleSyncDelayedTask(plugin, () -> {
            if (state == State.PLAYING) {
                endGame(null);
                plugin.getLogger().info("Game ended due to timeout");
            }
        }, MAX_DURATION);
    }

    public State getState() {
        return state;
    }

    public GameMap getMap() {
        return map;
    }

    public List<Player> getPlayers() {
        return players;
    }

    public void addPlayer(Player player) {
        if (state != State.WAITING_FOR_PLAYERS) throw new IllegalStateException("Session is in invalid state");
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
        if (state != State.WAITING_FOR_PLAYERS) throw new IllegalStateException("Session is in invalid state");
        if (players.size() != 2) throw new IllegalStateException("Need exactly 2 players to start a game");

        plugin.getLogger().info("Starting game");

        for (int i = 0; i < players.size(); i++) {
            Player player = players.get(i);

            player.teleport(spawnLocations[i]);
            lastLocations.put(player, spawnLocations[i]);

            player.sendMessage("Game started");
            player.setHealth(player.getMaxHealth());
            player.getInventory().clear();
        }

        state = State.PLAYING;
    }

    public void endGame(Player winner) {
        if (state == State.ENDED) throw new IllegalStateException("Game has already ended");

        state = State.ENDED;

        PlayerQuitEvent.getHandlerList().unregister(this);
        PlayerMoveEvent.getHandlerList().unregister(this);
        EntityDamageByEntityEvent.getHandlerList().unregister(this);

        if (winner == null) {
            for (Player player : players) {
                player.sendMessage("You lost");
            }
        } else {
            Player loser = getOtherPlayer(winner);
            loser.sendMessage("You lost");
            winner.sendMessage("You won");
        }

        sessionManager.removeSession(this);
        invisibilityManager.update();
    }

    public boolean hasPlayer(Player player) {
        return players.contains(player);
    }

    protected static GameMap pickMap(float randomizationFactor) {
        // when randomizationFactor 0 - always pick same map (White Crystal)
        // when randomizationFactor 1 - pick completely randomly

        if (random.nextDouble() < randomizationFactor) {
            return GameMap.values()[random.nextInt(GameMap.values().length)];
        }

        return GameMap.WHITE_CRYSTAL;
    }

    protected static int pickSpawnLocationPair(float randomizationFactor) {
        // when randomizationFactor 0 - always pick the first pair
        // when randomizationFactor 1 - pick completely randomly
        if (random.nextDouble() < randomizationFactor) {
            return random.nextInt(2);
        }

        return 0;
    }

    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        if (!hasPlayer(event.getPlayer()) || state != State.PLAYING) return;

        Player winner = getOtherPlayer(event.getPlayer());
        endGame(winner);

        plugin.getLogger().info("Game ended due to one player quiting");
    }

    @EventHandler
    public void onMove(PlayerMoveEvent event) {
        Player player = event.getPlayer();
        if (!hasPlayer(player) || state != State.PLAYING) return;

        if (event.getTo().getY() < map.minY) {
            Player winner = getOtherPlayer(player);
            endGame(winner);
            plugin.getLogger().info("Game ended due to one player falling off the platform");
        }
    }

    @EventHandler
    public void onPlayerDamage(EntityDamageByEntityEvent event) {
        if (!(event.getEntity() instanceof Player) || !(event.getDamager() instanceof Player) ||
            state != State.PLAYING) {
            return;
        }

        Player attacker = (Player) event.getDamager();
        Player target = (Player) event.getEntity();

        if (hasPlayer(attacker) && hasPlayer(target)) {
            sendExplorationReward(attacker, HIT_REWARD);
            sendExplorationReward(target, -HIT_REWARD);

            sendMetadata(attacker, "hit", 1);
        }
    }

    protected void sendExplorationReward(Player player, double reward) {
        player.sendMessage(String.format("Exploration reward %.5f", reward));
    }

    protected void sendMetadata(Player player, String key, double value) {
        player.sendMessage(String.format("metadata:%s:%.5f", key, value));
    }

    protected double distanceToCenter2d(Location location) {
        double dx = map.center.getX() - location.getX();
        double dz = map.center.getZ() - location.getZ();

        return Math.sqrt(dx * dx + dz * dz);
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
