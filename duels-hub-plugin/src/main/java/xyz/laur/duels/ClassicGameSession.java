package xyz.laur.duels;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.EntityType;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityShootBowEvent;
import org.bukkit.event.player.PlayerFishEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.plugin.Plugin;
import org.bukkit.projectiles.ProjectileSource;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class ClassicGameSession implements Listener, GameSession {
    public static final long MAX_DURATION = 3600; // 3 minutes

    public enum GameMap {
        ARENA(
                "Arena",
                new Location(null, 43.5, 71, -95.5, 0, 0),
                new Location(null, 43.5, 71, -27.5, -180, 0),
                new EmblemPosition[]{new EmblemPosition(43, 80, -96, true), new EmblemPosition(43, 80, -28, true)}
        ),
        BACKWOODS(
                "Backwoods",
                new Location(null, 4.5, 68, -226.5, -90, 0),
                new Location(null, 74.5, 68, -226.5, 90, 0),
                new EmblemPosition[]{new EmblemPosition(1, 85, -226, false), new EmblemPosition(78, 85, -226, false)}
        ),
        FIGHT_NIGHT(
                "Fight Night",
                new Location(null, -169.5, 44, -275.5, 0, 0),
                new Location(null, -169.5, 44, -205.5, -180, 0),
                new EmblemPosition[]{new EmblemPosition(-170, 66, -279, true), new EmblemPosition(-170, 66, -203, true)}
        ),
        FRACTAL(
                "Fractal",
                new Location(null, -182.5, 188, -266.5, 0, 0),
                new Location(null, -182.5, 188, -196.5, 180, 0),
                new EmblemPosition[]{new EmblemPosition(-183, 208, -271, true), new EmblemPosition(-183, 208, -193, true)}
        ),
        HIGHSET(
                "Highset",
                new Location(null, -159.5, 75, -11.5, -90, 0),
                new Location(null, -89.5, 75, -11.5, 90, 0),
                new EmblemPosition[]{new EmblemPosition(-164, 90, -12, false), new EmblemPosition(-87, 90, -12, false)}
        ),
        MUSEUM(
                "Museum",
                new Location(null, -292.5, 74, 7.5, -180, 0),
                new Location(null, -292.5, 74, -62.5, 0, 0),
                new EmblemPosition[]{new EmblemPosition(-293, 94, 10, true), new EmblemPosition(-293, 94, -65, true)}
        ),
        NEON(
                "Neon",
                new Location(null, 103.5, 48, 140.5, 90, 0),
                new Location(null, 33.5, 48, 140.5, -90, 0),
                new EmblemPosition[]{new EmblemPosition(106, 60, 140, false), new EmblemPosition(30, 60, 140, false)}
        ),
        REEF(
                "Reef",
                new Location(null, 189.5, 14, -109.5, 0, 0),
                new Location(null, 189.5, 14, -39.5, 180, 0),
                new EmblemPosition[]{new EmblemPosition(189, 40, -113, true), new EmblemPosition(189, 40, -37, true)}
        ),
        SKYPORT(
                "Skyport",
                new Location(null, -120.5, 28, 181.5, -180, 0),
                new Location(null, -120.5, 28, 111.5, 0, 0),
                new EmblemPosition[]{new EmblemPosition(-121, 44, 185, true), new EmblemPosition(-121, 44, 107, true)}
        ),
        SPIKEROCK_BAY(
                "Spikerock Bay",
                new Location(null, -262.5, 18, 176.5, 180, 0),
                new Location(null, -262.5, 18, 106.5, 0, 0),
                new EmblemPosition[]{new EmblemPosition(-264, 31, 179, true), new EmblemPosition(-264, 31, 103, true)}
        );

        private final String displayName;
        private final Location spawnLocation1;
        private final Location spawnLocation2;
        private final EmblemPosition[] emblemPositions;

        GameMap(String displayName, Location spawnLocation1, Location spawnLocation2,
                EmblemPosition[] emblemPositions) {
            this.displayName = displayName;
            this.spawnLocation1 = spawnLocation1;
            this.spawnLocation2 = spawnLocation2;
            this.emblemPositions = emblemPositions;
        }

        public static GameMap fromName(String name) {
            for (GameMap map : values()) {
                if (map.name().equalsIgnoreCase(name)) {
                    return map;
                }
            }

            throw new IllegalArgumentException("No map with name " + name);
        }

        public Location getSpawnLocation1() {
            return spawnLocation1;
        }

        public Location getSpawnLocation2() {
            return spawnLocation2;
        }

        public String getDisplayName() {
            return displayName;
        }

        public EmblemPosition[] getEmblemPositions() {
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
    private final float spawnDistance;
    private final Map<GameMap, MapBarrier> barriers;
    private final Location[] spawnLocations = new Location[2];
    private static final Random random = new Random();

    private int randomTeleportTask;

    public ClassicGameSession(World world, SessionManager sessionManager, InvisibilityManager invisibilityManager,
                              SkinChanger skinChanger, Plugin plugin, GameMap map, boolean randomTeleport,
                              float spawnDistance, Map<GameMap, MapBarrier> barriers) {
        this.sessionManager = sessionManager;
        this.invisibilityManager = invisibilityManager;
        this.skinChanger = skinChanger;
        this.plugin = plugin;
        this.map = map;
        this.randomTeleport = randomTeleport;
        this.spawnDistance = spawnDistance;
        this.barriers = barriers;

        setSpawnLocations(world, spawnDistance);

        plugin.getServer().getPluginManager().registerEvents(this, plugin);
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

        plugin.getLogger().info("Starting classic duels game");

        for (int i = 0; i < players.size(); i++) {
            Player player = players.get(i);
            player.teleport(spawnLocations[i]);

            player.sendMessage("Game started");
            player.setHealth(player.getMaxHealth());
            giveItems(player);
        }

        if (randomTeleport) {
            randomTeleportTask = plugin.getServer().getScheduler().scheduleSyncRepeatingTask(plugin,
                    this::randomTeleportTick, 40, 1);
        }

        // Postpone changing skin because sometimes it makes players invisible if set
        // at the same time as players are teleported
        plugin.getServer().getScheduler().scheduleSyncDelayedTask(plugin, () -> {
            if (state == GameState.PLAYING) {
                players.forEach(skinChanger::changeSkin);
            }
        }, 100);

        plugin.getServer().getScheduler().scheduleSyncDelayedTask(plugin, () -> {
            if (state == GameState.PLAYING) {
                endGame(null);
                plugin.getLogger().info("Classic duel game ended due to timeout");
            }
        }, MAX_DURATION);

        barriers.get(map).update(spawnDistance);

        state = GameState.PLAYING;
    }

    public void endGame(Player winner) {
        if (state == GameState.ENDED) throw new IllegalStateException("Game has already ended");

        state = GameState.ENDED;

        PlayerQuitEvent.getHandlerList().unregister(this);
        EntityDamageByEntityEvent.getHandlerList().unregister(this);
        EntityDamageEvent.getHandlerList().unregister(this);
        EntityShootBowEvent.getHandlerList().unregister(this);
        PlayerFishEvent.getHandlerList().unregister(this);

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

    protected void setSpawnLocations(World world, float spawnDistance) {
        if (random.nextBoolean()) {
            spawnLocations[0] = map.spawnLocation1.clone();
            spawnLocations[1] = map.spawnLocation2.clone();
        } else {
            spawnLocations[0] = map.spawnLocation2.clone();
            spawnLocations[1] = map.spawnLocation1.clone();
        }

        spawnLocations[0].setWorld(world);
        spawnLocations[1].setWorld(world);

        if (Math.abs(spawnLocations[0].getX() - spawnLocations[1].getX()) < 1) {
            double maxDistanceFromCenter = Math.abs(spawnLocations[0].getZ() - spawnLocations[1].getZ()) / 2;
            double offsetFromOriginal = (1 - spawnDistance) * maxDistanceFromCenter;

            if (spawnLocations[0].getZ() < spawnLocations[1].getZ()) {
                spawnLocations[0].setZ(spawnLocations[0].getZ() + offsetFromOriginal);
                spawnLocations[1].setZ(spawnLocations[1].getZ() - offsetFromOriginal);
            } else {
                spawnLocations[0].setZ(spawnLocations[0].getZ() - offsetFromOriginal);
                spawnLocations[1].setZ(spawnLocations[1].getZ() + offsetFromOriginal);
            }
        } else {
            double maxDistanceFromCenter = Math.abs(spawnLocations[0].getX() - spawnLocations[1].getX()) / 2;
            double offsetFromOriginal = (1 - spawnDistance) * maxDistanceFromCenter;

            if (spawnLocations[0].getX() < spawnLocations[1].getX()) {
                spawnLocations[0].setX(spawnLocations[0].getX() + offsetFromOriginal);
                spawnLocations[1].setX(spawnLocations[1].getX() - offsetFromOriginal);
            } else {
                spawnLocations[0].setX(spawnLocations[0].getX() - offsetFromOriginal);
                spawnLocations[1].setX(spawnLocations[1].getX() + offsetFromOriginal);
            }
        }
    }

    protected void giveItems(Player player) {
        player.getInventory().clear();

        ItemStack sword = new ItemStack(Material.IRON_SWORD);
        makeUnbreakable(sword);

        ItemStack bow = new ItemStack(Material.BOW);
        makeUnbreakable(bow);

        ItemStack fishingRod = new ItemStack(Material.FISHING_ROD);
        makeUnbreakable(fishingRod);

        player.getInventory().addItem(sword, bow, fishingRod);

        player.getInventory().setItem(9, new ItemStack(Material.ARROW, 5));

        ItemStack helmet = new ItemStack(Material.IRON_HELMET);
        helmet.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 2);
        makeUnbreakable(helmet);
        player.getInventory().setHelmet(helmet);

        ItemStack chestplate = new ItemStack(Material.IRON_CHESTPLATE);
        chestplate.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 2);
        makeUnbreakable(chestplate);
        player.getInventory().setChestplate(chestplate);

        ItemStack leggings = new ItemStack(Material.IRON_LEGGINGS);
        leggings.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 2);
        makeUnbreakable(leggings);
        player.getInventory().setLeggings(leggings);

        ItemStack boots = new ItemStack(Material.IRON_BOOTS);
        boots.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 2);
        makeUnbreakable(boots);
        player.getInventory().setBoots(boots);
    }

    protected void makeUnbreakable(ItemStack item) {
        ItemMeta meta = item.getItemMeta();
        meta.spigot().setUnbreakable(true);
        item.setItemMeta(meta);
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
        if (distanceBetweenPlayers < 3 && random.nextDouble() < 1D / 350) {
            plugin.getLogger().info("Randomly teleporting one player");
            Location newLocation = player.getLocation().clone();
            newLocation.setPitch(newLocation.getPitch() + (random.nextFloat() - 0.5F) * 90);
            newLocation.setYaw(newLocation.getYaw() + (random.nextFloat() - 0.5F) * 180);
            player.teleport(newLocation);
        }
    }

    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        if (!hasPlayer(event.getPlayer()) || state != GameState.PLAYING) return;

        Player winner = getOtherPlayer(event.getPlayer());
        endGame(winner);

        plugin.getLogger().info("Game ended due to one player quiting");
    }

    @EventHandler
    public void onBowShoot(EntityShootBowEvent event) {
        if (state != GameState.PLAYING) return;
        if (!(event.getEntity() instanceof Player)) return;

        Player shooter = (Player) event.getEntity();
        if (hasPlayer(shooter)) {
            sendMetadata(shooter, "shoot_arrow", event.getForce());
        }
    }

    @EventHandler
    public void onPlayerFish(PlayerFishEvent event) {
        if (state != GameState.PLAYING) return;
        if (event.getState() != PlayerFishEvent.State.FISHING) return;
        if (!hasPlayer(event.getPlayer())) return;

        sendMetadata(event.getPlayer(), "fishing_rod_used", 1);
    }

    @EventHandler
    public void onEntityDamage(EntityDamageEvent event) {
        if (state != GameState.PLAYING) return;
        if (!(event.getEntity() instanceof Player)) return;

        Player player = (Player) event.getEntity();

        if (hasPlayer(player) && player.getHealth() - event.getFinalDamage() <= 0) {
            event.setCancelled(true);
            endGame(getOtherPlayer(player));

            plugin.getLogger().info("Game ended due to one player dying");
        }
    }

    @EventHandler
    public void onPlayerDamage(EntityDamageByEntityEvent event) {
        if (!(event.getEntity() instanceof Player) || state != GameState.PLAYING) {
            return;
        }

        Player target = (Player) event.getEntity();

        if (!hasPlayer(target)) return;

        if (event.getDamager() instanceof Player) {
            Player attacker = (Player) event.getDamager();

            handleDamage(attacker, target, event.getFinalDamage());
        } else if (event.getDamager() instanceof Projectile) {
            ProjectileSource shooter = ((Projectile) event.getDamager()).getShooter();

            if (shooter instanceof Player) {
                Player attacker = (Player) shooter;

                if (hasPlayer(attacker)) {
                    if (event.getDamager().getType() == EntityType.FISHING_HOOK) {
                        sendFishingHookMetadata(attacker);
                    }

                    handleDamage(attacker, target, event.getFinalDamage());
                } else {
                    plugin.getLogger().info("Player got damaged by projectile from a player that is not in their session");
                    event.setCancelled(true);
                }
            }
        }
    }

    private void handleDamage(Player attacker, Player target, double damage) {
        if (hasPlayer(attacker) && hasPlayer(target) && damage > 0.1) {
            sendMetadata(attacker, "hits_done", damage);
            sendMetadata(target, "hits_received", damage);
        }
    }

    private void sendFishingHookMetadata(Player attacker) {
        sendMetadata(attacker, "fishing_hook_hit", 1);
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
