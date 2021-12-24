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
                        new Location[]{new Location(null, 127.5, 44, 93.5, -90, 0), new Location(null, 139.5, 44, 93.5, 90, 0)},
                        new Location[]{new Location(null, 133.5, 44, 99, -180, 0), new Location(null, 133.5, 44, 87.5, 0, 0)}
                },
                41
        ),
        SPACE_MINE(
                "Space Mine",
                new Location[][]{
                        new Location[]{new Location(null, 161.5, 144, 54.5, -90, 0), new Location(null, 173.5, 144, 54.5, 90, 0)},
                        new Location[]{new Location(null, 167.5, 144, 60.5, -180, 0), new Location(null, 167.5, 144, 48.5, 0, 0)}
                },
                141
        ),
        WHITE_CRYSTAL(
                "White Crystal",
                new Location[][]{
                        new Location[]{new Location(null, 214.5, 46, 169.5, -90, 0), new Location(null, 226.5, 46, 169.5, 90, 0)},
                        new Location[]{new Location(null, 220.5, 46, 175.5, 180, 0), new Location(null, 220.5, 46, 163.5, 0, 0)}
                },
                43
        ),
        PONSEN(
                "Ponsen",
                new Location[][]{
                        new Location[]{new Location(null, 216.5, 33, 29.5, -90, 0), new Location(null, 228.5, 33, 29.5, 90, 0)},
                        new Location[]{new Location(null, 222.5, 33, 35.5, 180, 0), new Location(null, 222.5, 33, 23.5, 0, 0)}
                },
                30
        ),
        FORT_ROYALE(
                "Fort Royale",
                new Location[][]{
                        new Location[]{new Location(null, 119.5, 44, 12.5, 90, 0), new Location(null, 107.5, 44, 12.5, -90, 0)},
                        new Location[]{new Location(null, 113.5, 44, 6.5, 0, 0), new Location(null, 113.5, 44, 18.5, 180, 0)}
                },
                41
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
