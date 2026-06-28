import { useState } from 'react';
import {
  AppBar,
  Box,
  Button,
  Container,
  Fab,
  Link,
  Toolbar,
  Typography,
} from '@mui/material';
import GraphicEqIcon from '@mui/icons-material/GraphicEq';
import BiotechIcon from '@mui/icons-material/Biotech';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import TesterDialog from './TesterDialog';

const APP_NAME = 'Foreshock';
const REPO_URL = 'https://github.com/FadyFaheem/Foreshock';

const NAV = [
  { label: 'Analyze', path: '/' },
  { label: 'Diagnostics', path: '/diagnostics' },
  { label: 'Health', path: '/health' },
  { label: 'Live', path: '/live' },
  { label: 'Fault Lab', path: '/fault-lab' },
];

// App shell: a top bar with navigation plus a scrollable main area. Foreshock
// is a small SPA, so this is a trimmed version of the template's auth-gated
// shell that keeps the same look and theme conventions.
export default function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const [testerOpen, setTesterOpen] = useState(false);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100dvh',
        width: '100%',
        overflow: 'hidden',
      }}
    >
      <AppBar position="static" sx={{ flexShrink: 0 }}>
        <Container maxWidth="xl">
          <Toolbar disableGutters>
            <GraphicEqIcon sx={{ mr: 1 }} />
            <Typography
              variant="h6"
              noWrap
              sx={{
                mr: 3,
                fontFamily: 'monospace',
                fontWeight: 700,
                letterSpacing: '.15rem',
                color: 'inherit',
              }}
            >
              {APP_NAME}
            </Typography>

            <Box sx={{ flexGrow: 1, display: 'flex', gap: 1 }}>
              {NAV.map((n) => {
                const active =
                  n.path === '/'
                    ? location.pathname === '/'
                    : location.pathname.startsWith(n.path);
                return (
                  <Button
                    key={n.path}
                    onClick={() => navigate(n.path)}
                    sx={{
                      color: 'primary.contrastText',
                      fontWeight: active ? 700 : 400,
                      borderBottom: '2px solid',
                      borderColor: active ? 'secondary.main' : 'transparent',
                      borderRadius: 0,
                    }}
                  >
                    {n.label}
                  </Button>
                );
              })}
            </Box>

            <Link
              href={REPO_URL}
              target="_blank"
              rel="noopener"
              underline="hover"
              sx={{ color: 'inherit', fontSize: 14 }}
            >
              GitHub
            </Link>
          </Toolbar>
        </Container>
      </AppBar>

      <Box
        component="main"
        sx={{
          flex: 1,
          minWidth: 0,
          overflow: 'auto',
          p: { xs: 1.5, sm: 2, md: 3 },
        }}
      >
        <Container maxWidth="xl" disableGutters>
          <Outlet />
        </Container>
      </Box>

      <Fab
        color="secondary"
        aria-label="signal tester"
        onClick={() => setTesterOpen(true)}
        sx={{ position: 'fixed', bottom: 24, right: 24 }}
      >
        <BiotechIcon />
      </Fab>
      <TesterDialog open={testerOpen} onClose={() => setTesterOpen(false)} />
    </Box>
  );
}
