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

    public enum State {
        WAITING_FOR_PLAYERS, PLAYING, ENDED
    }

    public enum GameMap {
        CLASSIC_SUMO(
                "Classic Sumo",
                new Location[][]{
                        new Location[]{new Location(null, -17.5, 77, 33.5, 90, 0), new Location(null, -29.5, 77, 33.5, -90, 0)},
                        new Location[]{new Location(null, -23.5, 77, 39.5, 180, 0), new Location(null, -23.5, 77, 27.5, 0, 0)}
                },
                73
        ),
        SPACE_MINE(
                "Space Mine",
                new Location[][]{
                        new Location[]{new Location(null, 101.5, 98, -188.5, 90, 0), new Location(null, 89.5, 98, -188.5, -90, 0)},
                        new Location[]{new Location(null, 95.5, 98, -182.5, -180, 0), new Location(null, 95.5, 98, -194.5, 0, 0)}
                },
                93
        ),
        WHITE_CRYSTAL(
                "White Crystal",
                new Location[][]{
                        new Location[]{new Location(null, 179.5, 80, -21.5, -90, 0), new Location(null, 191.5, 80, -21.5, 90, 0)},
                        new Location[]{new Location(null, 185.5, 80, -15.5, 180, 0), new Location(null, 185.5, 80, -27.5, 0, 0)}
                },
                74
        ),
        PONSEN(
                "Ponsen",
                new Location[][]{
                        new Location[]{new Location(null, 59.5, 67, -30.5, -90, 0), new Location(null, 71.5, 67, -30.5, 90, 0)},
                        new Location[]{new Location(null, 65.5, 67, -24.5, 180, 0), new Location(null, 65.5, 67, -36.5, 0, 0)}
                },
                60
        ),
        FORT_ROYALE(
                "Fort Royale",
                new Location[][]{
                        new Location[]{new Location(null, -45.5, 78, -74.5, -90, 0), new Location(null, -33.5, 78, -74.5, 90, 0)},
                        new Location[]{new Location(null, -39.5, 78, -68.5, 180, 0), new Location(null, -39.5, 78, -80.5, 0, 0)}
                },
                71
        );

        private final String displayName;
        private final Location[][] spawnLocations;
        private final double minY;

        GameMap(String displayName, Location[][] spawnLocations, double minY) {
            this.displayName = displayName;
            this.spawnLocations = spawnLocations;
            this.minY = minY;
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
            sendMetadata(attacker, "hit", 1);
            sendMetadata(target, "hit", -1);
        }
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
