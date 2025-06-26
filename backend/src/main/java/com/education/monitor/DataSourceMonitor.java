package com.education.monitor;

import com.zaxxer.hikari.HikariDataSource;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.SQLException;

/**
 * 数据库连接池监控组件
 * 提供连接池健康检查和状态监控功能
 * 
 * @author Education Platform Team
 * @since 1.0.0
 */
@Slf4j
@Component
public class DataSourceMonitor implements HealthIndicator {

    @Autowired
    private DataSource dataSource;

    /**
     * 健康检查实现
     * 用于Spring Boot Actuator健康检查端点
     * 
     * @return Health状态
     */
    @Override
    public Health health() {
        try {
            // 尝试获取连接来验证数据库可用性
            try (Connection connection = dataSource.getConnection()) {
                if (connection.isValid(5)) {
                    return buildHealthStatus(true);
                } else {
                    return Health.down()
                            .withDetail("error", "数据库连接无效")
                            .build();
                }
            }
        } catch (SQLException e) {
            log.error("数据库健康检查失败", e);
            return Health.down()
                    .withDetail("error", e.getMessage())
                    .withDetail("errorCode", e.getErrorCode())
                    .build();
        }
    }

    /**
     * 构建健康状态信息
     * 
     * @param isHealthy 是否健康
     * @return Health对象
     */
    private Health buildHealthStatus(boolean isHealthy) {
        Health.Builder builder = isHealthy ? Health.up() : Health.down();
        
        if (dataSource instanceof HikariDataSource) {
            HikariDataSource hikariDataSource = (HikariDataSource) dataSource;
            
            // 检查HikariPoolMXBean是否已初始化
            if (hikariDataSource.getHikariPoolMXBean() != null) {
                builder.withDetail("poolName", hikariDataSource.getPoolName())
                       .withDetail("activeConnections", hikariDataSource.getHikariPoolMXBean().getActiveConnections())
                       .withDetail("idleConnections", hikariDataSource.getHikariPoolMXBean().getIdleConnections())
                       .withDetail("totalConnections", hikariDataSource.getHikariPoolMXBean().getTotalConnections())
                       .withDetail("threadsAwaitingConnection", hikariDataSource.getHikariPoolMXBean().getThreadsAwaitingConnection())
                       .withDetail("maximumPoolSize", hikariDataSource.getMaximumPoolSize())
                       .withDetail("minimumIdle", hikariDataSource.getMinimumIdle());
            } else {
                builder.withDetail("poolName", hikariDataSource.getPoolName())
                       .withDetail("status", "连接池正在初始化中")
                       .withDetail("maximumPoolSize", hikariDataSource.getMaximumPoolSize())
                       .withDetail("minimumIdle", hikariDataSource.getMinimumIdle());
            }
        }
        
        return builder.build();
    }

    /**
     * 定期监控连接池状态
     * 每5分钟执行一次，记录连接池使用情况
     */
    @Scheduled(fixedRate = 300000) // 5分钟
    public void monitorConnectionPool() {
        if (dataSource instanceof HikariDataSource) {
            HikariDataSource hikariDataSource = (HikariDataSource) dataSource;
            
            // 检查HikariPoolMXBean是否已初始化
            if (hikariDataSource.getHikariPoolMXBean() == null) {
                log.warn("连接池监控跳过: HikariCP连接池尚未完全初始化");
                return;
            }
            
            try {
                int activeConnections = hikariDataSource.getHikariPoolMXBean().getActiveConnections();
                int idleConnections = hikariDataSource.getHikariPoolMXBean().getIdleConnections();
                int totalConnections = hikariDataSource.getHikariPoolMXBean().getTotalConnections();
                int threadsAwaitingConnection = hikariDataSource.getHikariPoolMXBean().getThreadsAwaitingConnection();
                
                log.info("=== 连接池监控报告 ===");
                log.info("连接池名称: {}", hikariDataSource.getPoolName());
                log.info("活跃连接: {}/{}", activeConnections, hikariDataSource.getMaximumPoolSize());
                log.info("空闲连接: {}", idleConnections);
                log.info("总连接数: {}", totalConnections);
                log.info("等待连接的线程: {}", threadsAwaitingConnection);
                
                // 连接池使用率告警
                double usageRate = (double) activeConnections / hikariDataSource.getMaximumPoolSize();
                if (usageRate > 0.8) {
                    log.warn("连接池使用率过高: {:.2f}%, 当前活跃连接: {}/{}", 
                            usageRate * 100, activeConnections, hikariDataSource.getMaximumPoolSize());
                }
                
                // 等待连接告警
                if (threadsAwaitingConnection > 0) {
                    log.warn("有 {} 个线程正在等待数据库连接，可能存在连接池不足的问题", threadsAwaitingConnection);
                }
            } catch (Exception e) {
                log.error("连接池监控过程中发生异常", e);
            }
        }
    }

    /**
     * 获取连接池详细信息
     * 用于管理接口或调试
     * 
     * @return 连接池信息字符串
     */
    public String getConnectionPoolInfo() {
        if (dataSource instanceof HikariDataSource) {
            HikariDataSource hikariDataSource = (HikariDataSource) dataSource;
            
            // 检查HikariPoolMXBean是否已初始化
            if (hikariDataSource.getHikariPoolMXBean() != null) {
                return String.format(
                    "连接池信息 - 名称: %s, 活跃: %d, 空闲: %d, 总计: %d, 最大: %d, 最小空闲: %d",
                    hikariDataSource.getPoolName(),
                    hikariDataSource.getHikariPoolMXBean().getActiveConnections(),
                    hikariDataSource.getHikariPoolMXBean().getIdleConnections(),
                    hikariDataSource.getHikariPoolMXBean().getTotalConnections(),
                    hikariDataSource.getMaximumPoolSize(),
                    hikariDataSource.getMinimumIdle()
                );
            } else {
                return String.format(
                    "连接池信息 - 名称: %s, 状态: 正在初始化中, 最大: %d, 最小空闲: %d",
                    hikariDataSource.getPoolName(),
                    hikariDataSource.getMaximumPoolSize(),
                    hikariDataSource.getMinimumIdle()
                );
            }
        }
        return "无法获取连接池信息";
    }
}