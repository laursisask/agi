package xyz.laur.duels;

import org.bukkit.World;
import org.bukkit.entity.Player;

public class InvisibilityManager {
    private final SessionManager sessionManager;
    private final World world;

    public InvisibilityManager(SessionManager sessionManager, World world) {
        this.sessionManager = sessionManager;
        this.world = world;
    }

    public void update() {
        for (Player a : world.getPlayers()) {
            for (Player b : world.getPlayers()) {
                GameSession sessionA = sessionManager.getPlayerSession(a);
                GameSession sessionB = sessionManager.getPlayerSession(b);
                boolean shouldSee = sessionA == null || sessionA.equals(sessionB);

                if (a.canSee(b) != shouldSee) {
                    if (shouldSee) {
                        a.showPlayer(b);
                    } else {
                        a.hidePlayer(b);
                    }
                }
            }
        }
    }
}
