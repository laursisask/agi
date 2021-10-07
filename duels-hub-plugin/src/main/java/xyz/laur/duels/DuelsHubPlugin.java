package xyz.laur.duels;

import org.bukkit.Bukkit;
import org.bukkit.World;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.FoodLevelChangeEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.weather.WeatherChangeEvent;
import org.bukkit.plugin.java.JavaPlugin;

public class DuelsHubPlugin extends JavaPlugin implements Listener {
    private InvisibilityManager invisibilityManager;

    @Override
    public void onEnable() {
        World world = Bukkit.getWorld("world");

        if (world == null) {
            throw new IllegalStateException("Could not find find game world");
        }

        world.setThundering(false);
        world.setStorm(false);

        SessionManager sessionManager = new SessionManager();
        invisibilityManager = new InvisibilityManager(sessionManager, world);
        invisibilityManager.update();

        getCommand("join").setExecutor(new JoinCommand(sessionManager, invisibilityManager, this, world));
        getCommand("games").setExecutor(new GamesCommand(sessionManager));

        getServer().getPluginManager().registerEvents(this, this);
    }

    @EventHandler
    public void onFoodLevelChange(FoodLevelChangeEvent event) {
        event.setCancelled(true);
    }

    @EventHandler
    public void onDamage(EntityDamageEvent event) {
        event.setDamage(0);
    }

    @EventHandler
    public void onBreakBlock(BlockBreakEvent event) {
        if (!event.getPlayer().isOp()) {
            event.setCancelled(true);
        }
    }

    @EventHandler
    public void onJoin(PlayerJoinEvent event) {
        invisibilityManager.update();
    }

    @EventHandler
    public void onWeatherChange(WeatherChangeEvent event) {
        if (event.toWeatherState()) {
            event.setCancelled(true);
        }
    }
}
