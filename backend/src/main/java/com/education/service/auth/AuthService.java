package com.education.service.auth;

import com.education.dto.AuthDTO;

public interface AuthService {
    
    AuthDTO.SimpleLoginResponse simpleLogin(AuthDTO.LoginRequest request);
    
    void simpleRegister(AuthDTO.RegisterRequest request);
    
    void changePassword(Long userId, AuthDTO.ChangePasswordRequest request);
} 